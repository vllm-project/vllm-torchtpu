# SPDX-License-Identifier: Apache-2.0
"""q38 L4 D98: direct cache-entry dispatch for the per-step compiled helpers.

Every decode step dispatches a dozen ``torch.compile(backend="tpu",
fullgraph=True, dynamic=False)`` helpers. Each call pays Dynamo's frame:
``compile_wrapper`` / ``_fn``, the config context, the eval-frame callback,
the C cache lookup and guard evaluation, then the transformed frame (MM FD-H,
``mm/reports/lane_host.md``: about 85-105 us of a 294-318 us dispatch). This
module skips everything above the transformed frame:

* ``VLLM_TPU_FAST_COMPILED_CALL=1``: ``torch.compile`` is hooked so that an
  eligible plain function (backend "tpu", fullgraph, dynamic=False, no
  options / mode, defined in vllm_torchtpu) comes back wrapped. The wrapper
  evaluates the guard manager of each of its own backend's cache entries in
  MRU order against the call's locals and runs the hit entry's transformed
  code object directly. On no hit, on any unusual state (a Dynamo callback or
  non-default stance active, tracing) or any error, it makes the normal call.
  It never compiles anything itself: a program is compiled only by the normal
  call, exactly when the unwrapped function would compile it.
* ``VLLM_TPU_FAST_CALL_GUARD_MEMO=1``: on top, remember which entry the guard
  check picked for an argument signature (per tensor: type, shape, stride,
  storage_offset, dtype, device, requires_grad, is_inference, object-identity
  aliasing; scalars by type and value; other objects by id; closure cells the
  same way) under a snapshot of the global state, and dispatch later calls
  with the same signature to it without running the guards. torch_tpu's
  compiled consumers specialise on storage_offset, so it is in the key. The
  memo is dropped whenever the code object's cache entries change (recompile,
  eviction, reset) or the global state differs from its snapshot, and it is
  never used for an entry whose guards read attributes of an argument (a
  mutable ``self``) or compare storages. ``=check`` runs the real guards on
  every memo hit, dispatches the guard-selected entry and counts the hits whose
  memoized entry fails its guards (``memo_mismatch``).
* ``VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE=1``: compile the eligible helpers with
  a ``guard_filter_fn`` that drops only the GLOBAL_STATE guard (grad mode,
  num_threads, autocast, ...), so warm-up entries are reused when the serving
  thread's global state differs. The graphs are unchanged.
* ``VLLM_TPU_FAST_CALL_LOG_EVERY=N``: with the fast call on, each rank logs a
  ``D98FAST`` engagement line every N runner steps (``step_tick``).

All flags are host-only and listed in ``_TPU_COMPILE_ENV_IGNORED``.

Credit: IX ``QX_FAST_COMPILED_CALL``, ``VLLM_TPU_T3_GUARD_MEMO`` and
``QX_TPU_DROP_GLOBAL_STATE_GUARD`` (ix scratch/l7b/treeA
src/vllm_torchtpu/_qx_fastcall.py); MM FD-H option D measurements.
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
import types
from typing import Any

import torch
import torch._dynamo.eval_frame as _ef
from torch._C._dynamo import eval_frame as _E

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Backends whose helpers are wrapped (tests add CPU stand-ins).
ELIGIBLE_BACKENDS: tuple[Any, ...] = ("tpu",)
# Only functions defined in these packages are wrapped.
MODULE_PREFIXES: tuple[str, ...] = ("vllm_torchtpu",)

STATS: dict[str, int] = {
    "fast": 0,  # calls dispatched to a transformed code object directly
    "slow": 0,  # calls that took the normal compiled call
    "memo_hit": 0,
    "memo_store": 0,
    "memo_check_ok": 0,
    "memo_mismatch": 0,
    "memo_nokey": 0,  # arguments not memoizable (kwargs, markers, deep seqs)
    "memo_unsafe": 0,  # entry guards read argument attributes / storages
}
_WRAPPED: list[dict[str, Any]] = []
_MEMO_MODE = ""  # "", "1" or "check"
_MEMO_MAX_ENTRIES = 64
_MEMO_MAX_SEQ = 16
_MISMATCH_LOG_LIMIT = 20
_SCALARS = frozenset((bool, int, float, str, type(None), torch.dtype, torch.device))
_DYN_ATTRS = (
    "_dynamo_dynamic_indices",
    "_dynamo_weak_dynamic_indices",
    "_dynamo_unbacked_indices",
    "_dynamo_strict_unbacked_indices",
    "_dynamo_static_indices",
)
# A guard on an attribute of a local (``L['self'].n == 3``) or a storage
# comparison makes an entry unsafe to memoize on argument signatures.
_UNSAFE_PART = re.compile(r"L\['[^']*'\](?:\[[^\]]*\])*\.|check_overlapping")

_INSTALLED = False
_LOG_EVERY = 0
_RANK = -1
_STEPS = 0
_LAST: dict[str, int] = {}
_MISMATCH_LOGGED = 0


def parse_memo_mode(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in ("", "0", "false", "off", "no"):
        return ""
    return "check" if v == "check" else "1"


def set_rank(rank: int) -> None:
    global _RANK
    _RANK = rank


def stats() -> dict[str, int]:
    return dict(STATS)


def _reset_stats_for_tests() -> None:
    global _STEPS, _MISMATCH_LOGGED
    for k in STATS:
        STATS[k] = 0
    _LAST.clear()
    _STEPS = 0
    _MISMATCH_LOGGED = 0


def _find_backend(compiled: Any) -> Any:
    """The backend object Dynamo stores on this compile's cache entries."""
    code = getattr(compiled, "__code__", None)
    cells = getattr(compiled, "__closure__", None) or ()
    if code is None:
        return None
    named = dict(zip(code.co_freevars, cells))
    ordered = [named["self"]] if "self" in named else []
    ordered += [c for c in cells if c not in ordered]
    for cell in ordered:
        try:
            obj = cell.cell_contents
        except ValueError:
            continue
        cb = getattr(obj, "callback", None)
        if cb is not None and hasattr(obj, "__enter__"):
            return _ef.innermost_backend(cb)
    return None


def _has_dyn_attr(d: dict) -> bool:
    for name in _DYN_ATTRS:
        if name in d:
            return True
    return False


def _key_into(parts: list, seq: Any, seen: dict) -> bool:
    for a in seq:
        t = type(a)
        if t in _SCALARS:
            parts.append((t, a))
        elif isinstance(a, torch.Tensor):
            d = a.__dict__
            if d and _has_dyn_attr(d):
                return False
            i = id(a)
            first = seen.get(i)
            if first is not None:
                parts.append(("alias", first))
                continue
            seen[i] = len(parts)
            parts.append(
                (
                    t,
                    a.shape,
                    a.stride(),
                    a.storage_offset(),
                    a.dtype,
                    a.device,
                    a.requires_grad,
                    a.is_inference(),
                )
            )
        elif t is tuple or t is list:
            if len(a) > _MEMO_MAX_SEQ:
                return False
            parts.append((t, len(a)))
            if not _key_into(parts, a, seen):
                return False
        else:
            parts.append(("id", id(a)))
    return True


def memo_key(args: tuple, cells: tuple = (), kwargs: dict | None = None) -> tuple | None:
    """Argument signature for the guard memo, or None (not memoizable).

    Keyword arguments enter by name in sorted order; closure cells by their
    current contents (a rebound cell changes the key)."""
    parts: list = []
    seen: dict = {}
    if not _key_into(parts, args, seen):
        return None
    if kwargs:
        names = sorted(kwargs)
        parts.append(("kw", tuple(names)))
        if not _key_into(parts, [kwargs[n] for n in names], seen):
            return None
    if cells:
        parts.append("cells")
        try:
            vals = tuple(c.cell_contents for c in cells)
        except ValueError:
            return None
        if not _key_into(parts, vals, seen):
            return None
    return tuple(parts)


def _guard_parts(gm: Any) -> list[str]:
    parts = getattr(gm, "code_parts", None)
    if parts is None:
        return []
    return [str(p) for p in parts]


def _memo_safe(gm: Any) -> bool:
    parts = getattr(gm, "code_parts", None)
    if parts is None:
        return False
    return not any(_UNSAFE_PART.search(str(p)) for p in parts)


def _entry_count(code: types.CodeType) -> int:
    return _E._get_total_cache_entry_count(code)


def _region_entries(code: types.CodeType, region: int) -> list:
    getter = getattr(_E, "_get_cache_entries_for_region", None)
    if getter is not None:
        return list(getter(code, region))
    return list(_E._debug_get_cache_entry_list(code))


def _head_code(code: types.CodeType) -> Any:
    entries = _E._debug_get_cache_entry_list(code)
    return entries[0].code if entries else None


def _wrap(compiled: Any, fn: types.FunctionType) -> Any:
    """Wrap one compiled plain function (see the module docstring)."""
    code = fn.__code__
    flags = code.co_flags
    has_varargs = bool(flags & inspect.CO_VARARGS)
    simple_sig = not (flags & inspect.CO_VARKEYWORDS)
    nargs = code.co_argcount
    nkw = code.co_kwonlyargcount
    nsig = nargs + nkw + (1 if has_varargs else 0)
    arg_names = code.co_varnames[:nargs]
    sig_names = frozenset(code.co_varnames[: nargs + nkw])
    var_name = code.co_varnames[nargs + nkw] if has_varargs else None
    defaults: dict[str, Any] = {}
    if fn.__defaults__:
        defaults.update(zip(arg_names[nargs - len(fn.__defaults__) :], fn.__defaults__))
    if fn.__kwdefaults__:
        defaults.update(fn.__kwdefaults__)
    freevars = code.co_freevars
    closure = fn.__closure__ or ()
    sig = inspect.signature(fn)
    region = getattr(compiled, "_isolate_recompiles_id", -1)
    if region is None:
        region = -1
    backend = _find_backend(compiled)
    st: dict[str, Any] = {
        "name": f"{fn.__module__}.{fn.__qualname__}",
        "entries": [],  # MRU list of (guard_manager, direct_fn, memo_safe)
        "count": -1,  # total cache entries when ``entries`` was built
        "head": None,  # code object of Dynamo's head entry then
        "disabled": "" if simple_sig else "**kwargs signature",
        "logged": False,
        "memo": {},  # signature -> direct fn (current entry generation)
        "memo_gs": None,  # GlobalStateGuard snapshot the memo is valid under
        "memo_tf": 0,
        "fast": 0,
        "slow": 0,
    }
    if backend is None:
        st["disabled"] = "backend not found"

    def _f_locals_bind(args: tuple, kwargs: dict) -> dict | None:
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError:
            return None
        bound.apply_defaults()
        return dict(bound.arguments)

    def _f_locals(args: tuple, kwargs: dict) -> dict | None:
        """The frame's locals as Dynamo's guards see them (``L[...]``)."""
        n = len(args)
        if n == nargs and not kwargs and not nkw:
            f_locals = dict(zip(arg_names, args))
            if has_varargs:
                f_locals[var_name] = ()
        elif n > nargs and has_varargs and not kwargs and not nkw:
            f_locals = dict(zip(arg_names, args))
            f_locals[var_name] = args[nargs:]
        elif n <= nargs:
            f_locals = dict(zip(arg_names, args))
            for k, v in kwargs.items():
                if k in f_locals or k not in sig_names:
                    return _f_locals_bind(args, kwargs)
                f_locals[k] = v
            if len(f_locals) < nargs + nkw:
                for name in code.co_varnames[: nargs + nkw]:
                    if name not in f_locals:
                        if name not in defaults:
                            return None
                        f_locals[name] = defaults[name]
            if has_varargs:
                f_locals[var_name] = ()
        else:
            f_locals = _f_locals_bind(args, kwargs)
            if f_locals is None:
                return None
        try:
            for name, cell in zip(freevars, closure):
                f_locals[name] = cell.cell_contents
        except ValueError:  # an empty cell
            return None
        return f_locals

    def _clear() -> None:
        st["entries"] = []
        st["count"] = -1
        st["head"] = None
        st["memo"] = {}
        st["memo_gs"] = None

    def _refresh(f_locals: dict | None) -> None:
        """After a normal call: rebuild this backend's entries if the entry
        generation changed (never compiles; a frame no entry accepts, e.g. a
        recompile-limit eager fallback, just keeps taking the normal call)."""
        try:
            count = _entry_count(code)
            head = _head_code(code)
            if st["entries"] and count == st["count"] and head is st["head"]:
                return
            _clear()
            built: list[tuple[Any, Any, bool]] = []
            for entry in _region_entries(code, region):
                if entry.backend is not backend:
                    continue
                ecode = entry.code
                if (
                    ecode.co_freevars != freevars
                    or ecode.co_argcount != nargs
                    or ecode.co_kwonlyargcount != nkw
                    or (ecode.co_flags ^ flags)
                    & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
                    or ecode.co_varnames[:nsig] != code.co_varnames[:nsig]
                ):
                    st["disabled"] = "transformed code signature differs"
                    logger.info(
                        "D98 fast call: %s disabled (%s)", st["name"], st["disabled"]
                    )
                    return
                direct = types.FunctionType(
                    ecode, fn.__globals__, fn.__name__, fn.__defaults__,
                    fn.__closure__,
                )
                direct.__kwdefaults__ = fn.__kwdefaults__
                gm = entry.guard_manager
                built.append((gm, direct, _memo_safe(gm)))
            if f_locals is not None:
                for idx, (gm, _direct, _safe) in enumerate(built):
                    if gm.check(f_locals):
                        if idx:
                            built.insert(0, built.pop(idx))
                        break
            st["entries"] = built
            st["count"] = count
            st["head"] = head
        except Exception:  # noqa: BLE001 - fall back to the normal call for good
            _clear()
            st["disabled"] = "refresh error"
            logger.warning("D98 fast call: %s disabled", st["name"], exc_info=True)

    def _memo_reset() -> None:
        st["memo"] = {}
        st["memo_gs"] = torch._C._dynamo.guards.GlobalStateGuard()
        st["memo_tf"] = torch._C._len_torch_function_stack()

    def _memo_valid() -> bool:
        """The memo belongs to the current entry generation (``_clear`` drops
        it on every refresh); it is also void when the global state or the
        torch-function mode stack differs from its snapshot."""
        gs = st["memo_gs"]
        if (
            gs is None
            or not gs.check()
            or st["memo_tf"] != torch._C._len_torch_function_stack()
        ):
            _memo_reset()
            return False
        return True

    def _memo_lookup(args: tuple, kwargs: dict) -> Any:
        """The direct fn to call for this signature, or None."""
        global _MISMATCH_LOGGED
        if not _memo_valid():
            return None
        key = memo_key(args, closure, kwargs)
        if key is None:
            STATS["memo_nokey"] += 1
            return None
        hit = st["memo"].get(key)
        if hit is None:
            return None
        if _MEMO_MODE == "check":
            f_locals = _f_locals(args, kwargs)
            sel = None
            ok = False
            for gm, direct, _safe in st["entries"]:
                accepted = gm.check(f_locals)
                if accepted and sel is None:
                    sel = direct
                if direct is hit and accepted:
                    ok = True
            if ok:
                STATS["memo_check_ok"] += 1
            else:
                STATS["memo_mismatch"] += 1
                if _MISMATCH_LOGGED < _MISMATCH_LOG_LIMIT:
                    _MISMATCH_LOGGED += 1
                    logger.warning(
                        "D98 guard memo check: memoized entry of %s fails its "
                        "guards (guard-selected: %s)",
                        st["name"],
                        "none" if sel is None else "another entry",
                    )
            if sel is None:
                return None
            hit = sel
        STATS["memo_hit"] += 1
        return hit

    def _memo_store(args: tuple, kwargs: dict, direct: Any, safe: bool) -> None:
        if not safe:
            STATS["memo_unsafe"] += 1
            return
        memo = st["memo"]
        if len(memo) >= _MEMO_MAX_ENTRIES:
            return
        key = memo_key(args, closure, kwargs)
        if key is None:
            STATS["memo_nokey"] += 1
            return
        if key not in memo:
            memo[key] = direct
            STATS["memo_store"] += 1

    def _fast(direct: Any, args: tuple, kwargs: dict) -> Any:
        STATS["fast"] += 1
        st["fast"] += 1
        if not st["logged"]:
            st["logged"] = True
            logger.info("D98 fast call: first direct dispatch of %s", st["name"])
        return direct(*args, **kwargs)

    def _select(args: tuple, kwargs: dict) -> tuple[Any, dict | None]:
        """(direct fn to run, or None; the call's locals if built)."""
        entries = st["entries"]
        if not (
            entries
            and _E.get_eval_frame_callback() is None
            and _ef._stance.stance == "default"
            and not torch.compiler.is_compiling()
            # Same entry generation: no recompile, eviction or reset since
            # ``entries`` was built (a new entry always lands at the head).
            and _entry_count(code) == st["count"]
            and _head_code(code) is st["head"]
        ):
            return None, None
        memo_ok = bool(_MEMO_MODE)
        if memo_ok and st["memo"]:
            hit = _memo_lookup(args, kwargs)
            if hit is not None:
                return hit, None
        f_locals = _f_locals(args, kwargs)
        if f_locals is None:
            return None, None
        for idx, (gm, direct, safe) in enumerate(entries):
            if gm.check(f_locals):
                if memo_ok:
                    _memo_valid()
                    _memo_store(args, kwargs, direct, safe)
                if idx:
                    # Copy-on-write MRU update (no in-place mutation of a list
                    # another call may be iterating).
                    st["entries"] = [entries[idx]] + entries[:idx] + entries[idx + 1 :]
                return direct, f_locals
        return None, f_locals

    @functools.wraps(fn)
    def d98_fast_compiled(*args: Any, **kwargs: Any) -> Any:
        if st["disabled"]:
            STATS["slow"] += 1
            st["slow"] += 1
            return compiled(*args, **kwargs)
        try:
            direct, f_locals = _select(args, kwargs)
            if f_locals is None and direct is None:
                f_locals = _f_locals(args, kwargs)
        except Exception:  # noqa: BLE001 - never fail a call the normal path runs
            _clear()
            st["disabled"] = "select error"
            logger.warning("D98 fast call: %s disabled", st["name"], exc_info=True)
            direct, f_locals = None, None
        if direct is not None:
            return _fast(direct, args, kwargs)
        out = compiled(*args, **kwargs)
        STATS["slow"] += 1
        st["slow"] += 1
        if not st["disabled"]:
            _refresh(f_locals)
        return out

    d98_fast_compiled._d98_fast_state = st  # type: ignore[attr-defined]
    # Inside a traced region Dynamo inlines the original function, as it does
    # for the compile_wrapper this replaces.
    d98_fast_compiled._torchdynamo_inline = getattr(  # type: ignore[attr-defined]
        compiled, "_torchdynamo_inline", fn
    )
    d98_fast_compiled._torchdynamo_orig_callable = getattr(  # type: ignore[attr-defined]
        compiled, "_torchdynamo_orig_callable", fn
    )
    d98_fast_compiled._torchdynamo_wrapper_id = id(d98_fast_compiled)  # type: ignore[attr-defined]
    d98_fast_compiled._isolate_recompiles_id = region  # type: ignore[attr-defined]
    d98_fast_compiled._d98_compiled = compiled  # type: ignore[attr-defined]
    _WRAPPED.append(st)
    return d98_fast_compiled


def _drop_global_state(entries: Any) -> list[bool]:
    return [e.guard_type != "GLOBAL_STATE" for e in entries]


def _eligible(model: Any, kwargs: dict) -> bool:
    return (
        isinstance(model, types.FunctionType)
        and kwargs.get("backend") in ELIGIBLE_BACKENDS
        and kwargs.get("fullgraph") is True
        and kwargs.get("dynamic") is False
        and not kwargs.get("disable", False)
        and kwargs.get("options") is None
        and kwargs.get("mode") is None
        and (model.__module__ or "").startswith(MODULE_PREFIXES)
    )


def make_hook(base: Any, fast: bool, drop_global_state: bool) -> Any:
    """A ``torch.compile`` replacement around ``base`` (idempotent result)."""

    @functools.wraps(base)
    def d98_compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
        if args:
            return base(model, *args, **kwargs)
        if model is None:

            def deco(m: Any) -> Any:
                return d98_compile(m, **kwargs)

            return deco
        if not _eligible(model, kwargs):
            return base(model, **kwargs)
        ckw = dict(kwargs)
        if drop_global_state:
            # torch.compile pops guard_filter_fn; the emptied dict is falsy,
            # so the backend still gets no options.
            ckw["options"] = {"guard_filter_fn": _drop_global_state}
        out = base(model, **ckw)
        if fast and callable(out) and not hasattr(out, "_d98_fast_state"):
            return _wrap(out, model)
        return out

    d98_compile._d98_hook = True  # type: ignore[attr-defined]
    return d98_compile


def install(
    fast: bool = True,
    drop_global_state: bool = False,
    memo: str = "",
    log_every: int = 0,
) -> bool:
    """Hook ``torch.compile`` (and torch_tpu's captured copy). Idempotent."""
    global _INSTALLED, _MEMO_MODE, _LOG_EVERY
    _MEMO_MODE = parse_memo_mode(memo) if fast else ""
    _LOG_EVERY = max(0, int(log_every))
    if not (fast or drop_global_state):
        return False
    changed = False
    if not getattr(torch.compile, "_d98_hook", False):
        torch.compile = make_hook(torch.compile, fast, drop_global_state)
        changed = True
    # torch_tpu captures torch.compile at import and re-installs its own
    # wrapper around that copy at device init; hook the copy too, so the order
    # of torch_tpu's import, its device init and this install does not matter.
    loader = sys.modules.get("torch_tpu._loader")
    base = getattr(loader, "_torch_compile", None) if loader is not None else None
    if base is not None and not getattr(base, "_d98_hook", False):
        loader._torch_compile = make_hook(base, fast, drop_global_state)
        changed = True
    if changed or not _INSTALLED:
        logger.info(
            "D98 fast compiled call installed: fast=%s memo=%s drop_global_state=%s "
            "log_every=%d torch_tpu_loader=%s",
            fast,
            _MEMO_MODE or "off",
            drop_global_state,
            _LOG_EVERY,
            loader is not None,
        )
    _INSTALLED = True
    return True


def wrapped_states() -> list[dict[str, Any]]:
    return list(_WRAPPED)


def engagement_line() -> str:
    s = STATS
    tot = s["fast"] + s["slow"]
    win = {k: s[k] - _LAST.get(k, 0) for k in ("fast", "slow", "memo_hit", "memo_mismatch")}
    win_tot = win["fast"] + win["slow"]
    engaged = sum(1 for st in _WRAPPED if st["fast"])
    slow_top = sorted(
        ((st["slow"] - st.get("_last_slow", 0), st["name"]) for st in _WRAPPED),
        reverse=True,
    )[:4]
    for st in _WRAPPED:
        st["_last_slow"] = st["slow"]
    top = ",".join(f"{n.rsplit('.', 1)[-1]}:{c}" for c, n in slow_top if c)
    disabled = ",".join(
        f"{st['name'].rsplit('.', 1)[-1]}({st['disabled']})"
        for st in _WRAPPED
        if st["disabled"] and st["slow"]
    )
    line = (
        f"D98FAST rank={_RANK} steps={_STEPS} fast={s['fast']} slow={s['slow']} "
        f"fast_share={(s['fast'] / tot) if tot else 0.0:.4f} "
        f"memo={_MEMO_MODE or 'off'} memo_hit={s['memo_hit']} "
        f"memo_store={s['memo_store']} memo_check_ok={s['memo_check_ok']} "
        f"memo_mismatch={s['memo_mismatch']} memo_nokey={s['memo_nokey']} "
        f"memo_unsafe={s['memo_unsafe']} wrapped={len(_WRAPPED)} engaged={engaged} "
        f"win_fast={win['fast']} win_slow={win['slow']} "
        f"win_fast_share={(win['fast'] / win_tot) if win_tot else 0.0:.4f} "
        f"win_memo_hit={win['memo_hit']} win_memo_mismatch={win['memo_mismatch']} "
        f"win_slow_top=[{top}] disabled=[{disabled}]"
    )
    _LAST.update({k: s[k] for k in ("fast", "slow", "memo_hit", "memo_mismatch")})
    return line


def step_tick() -> None:
    """Once per runner step: log the engagement line every N steps."""
    global _STEPS
    _STEPS += 1
    if _LOG_EVERY and _STEPS % _LOG_EVERY == 0:
        logger.info("%s", engagement_line())
