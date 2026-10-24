# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Guard-checked direct dispatch of this package's compiled TPU functions.

``compilation.decorators.tpu_compile`` compiles a function with
``torch.compile(backend="tpu", fullgraph=True, dynamic=False)`` and calls
:func:`wrap` on the result. The wrapper calls the compiled function and
nothing else until the worker arms ``fast_compiled_call`` from
``additional_config["tpu_compile"]`` (``compilation.compile_options``; off by
default). It reads the option on every call, so a function decorated at
import, before the options are known, is covered. Armed, it evaluates each
cached entry's ``guard_manager.check(f_locals)`` in MRU order and invokes the
transformed Python code object directly on a guard hit, skipping Dynamo's
per-call frame evaluation; every guard Dynamo built is kept.

* ``VLLM_TPU_T3_GUARD_MEMO=1`` (default off): on top of the direct dispatch,
  remember which cache entry the guard check selected for an argument
  *signature* (per tensor: shape, stride, dtype, device, requires_grad; per
  Python scalar / None / str: its type and value; any other object: its id;
  lists and tuples element-wise) and dispatch later calls with the same
  signature to that entry without re-running ``guard_manager.check``. The
  memo is dropped whenever the code object's cache entries change (recompile
  or eviction). ``VLLM_TPU_T3_GUARD_MEMO=check`` still runs the guard check on
  every memo hit, dispatches the guard-selected entry, and counts calls where
  the memo would have picked an entry whose guards fail (logged).

TODO: Remove this module once PyTorch has a public guarded call path for
``fullgraph`` compiled functions (https://github.com/pytorch/pytorch/issues/206560);
``tpu_compile`` then calls that path instead of :func:`wrap`. Until then it
relies on Dynamo internals, checked against torch 2.13.0 (git cf30153c4c13):
``torch._C._dynamo.eval_frame._get_cache_entries_for_region`` and
``_get_total_cache_entry_count``, each cache entry's ``code`` and
``guard_manager.check`` and ``cache_entry`` (None once Dynamo invalidates or
frees the entry), the compiled wrapper's ``_isolate_recompiles_id``,
``torch._dynamo.eval_frame._stance``, and calling an entry's transformed code
object directly. A torch upgrade must re-check them.
"""

from __future__ import annotations

import functools
import inspect
import os
import types
from typing import Any

import torch
from torch._C._dynamo import eval_frame as _c_eval_frame

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# [fast, slow] calls, counted over every wrapped function of the process;
# logged at INFO every _LOG_EVERY calls so serving logs show the hit rate.
_STATS = [0, 0]
# VLLM_TPU_T3_GUARD_MEMO: "" / "0" off, "1" on, "check" on with verification.
def _parse_memo_mode(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in ("", "0", "false", "off", "no"):
        return ""
    return "check" if v == "check" else "1"


# VLLM_TPU_T3_HOST=1 is the master switch for the IX track-3 host trims; an
# explicit VLLM_TPU_T3_GUARD_MEMO wins over it.
_MEMO_MODE = _parse_memo_mode(
    os.environ.get("VLLM_TPU_T3_GUARD_MEMO", os.environ.get("VLLM_TPU_T3_HOST"))
)
_MEMO_STATS = {"hit": 0, "store": 0, "check_ok": 0, "check_mismatch": 0, "nokey": 0}
_MEMO_MAX_ENTRIES = 64
_MEMO_MAX_SEQ = 16
_SCALARS = (bool, int, float, str)


def _memo_key(args: tuple | list) -> tuple | None:
    """Cheap argument signature for the guard memo, or None (not memoizable).

    Covers everything Dynamo's argument guards can see on these call sites for
    a fixed specialization: tensor metadata (not values: tensor values are
    never guarded), scalar values, and object identity for anything else
    (``self`` of a compiled method, modules, owners)."""
    parts: list[Any] = []
    for a in args:
        if isinstance(a, torch.Tensor):
            parts.append((a.shape, a.stride(), a.dtype, a.device, a.requires_grad))
        elif a is None or type(a) in _SCALARS:
            parts.append((type(a), a))
        elif type(a) in (list, tuple):
            if len(a) > _MEMO_MAX_SEQ:
                return None
            sub = _memo_key(a)
            if sub is None:
                return None
            parts.append((type(a), sub))
        else:
            parts.append(("id", id(a)))
    return tuple(parts)


def memo_stats() -> dict[str, int]:
    return dict(_MEMO_STATS)
_LOG_EVERY = 20000
# Armed by configure(); off until the worker applies tpu_compile.
_CONFIG = {"fast": False}


def configure(fast: bool) -> None:
    """Arm (or disarm) the direct dispatch of every ``tpu_compile`` function
    of the process. The worker calls this before any compiled
    function runs; the wrapper checks it on every call."""
    _CONFIG["fast"] = bool(fast)
    if fast:
        logger.info(
            "tpu_compile: fast_compiled_call=True for the package's "
            "tpu_compile functions"
        )


def _region_entries(code: types.CodeType, region: int) -> list:
    return list(_c_eval_frame._get_cache_entries_for_region(code, region))


def _entry_count(code: types.CodeType) -> int:
    return _c_eval_frame._get_total_cache_entry_count(code)


def wrap(compiled: Any, fn: types.FunctionType) -> Any:
    """The fast-dispatch wrapper of ``compiled``, ``torch.compile``'s result
    for the plain function ``fn``."""
    code = fn.__code__
    region = compiled._isolate_recompiles_id
    flags = code.co_flags
    simple_sig = not (flags & inspect.CO_VARARGS or flags & inspect.CO_VARKEYWORDS)
    nargs = code.co_argcount
    arg_names = code.co_varnames[:nargs]
    # Keyword-only defaults: a positional call leaves them out of args.
    kwdefaults = fn.__kwdefaults__
    freevars = code.co_freevars
    closure = fn.__closure__
    sig = inspect.signature(fn)
    stance_mod = torch._dynamo.eval_frame
    state: dict[str, Any] = {
        "entries": [],  # MRU list of (guard_manager, direct_fn)
        "count": -1,
        "disabled": not simple_sig,
        "logged": False,
        "memo": {},  # signature -> direct fn (VLLM_TPU_T3_GUARD_MEMO)
        "memo_count": -1,
    }

    def _f_locals(args: tuple, kwargs: dict) -> dict | None:
        if kwargs or len(args) != nargs:
            try:
                bound = sig.bind(*args, **kwargs)
            except TypeError:
                return None
            bound.apply_defaults()
            f_locals = dict(bound.arguments)
        else:
            f_locals = dict(zip(arg_names, args))
            if kwdefaults:
                f_locals.update(kwdefaults)
        if freevars:
            # A function with free variables always has a closure.
            for name, cell in zip(freevars, closure or ()):
                f_locals[name] = cell.cell_contents
        return f_locals

    def _refresh(f_locals: dict | None) -> None:
        state["memo"] = {}
        state["entries"] = []
        state["count"] = -1
        if f_locals is None:
            return
        try:
            entries = _region_entries(code, region)
            if not entries:
                return
            built: list[tuple[Any, Any]] = []
            mru_idx = -1
            for entry in entries:
                ecode = entry.code
                if ecode is None:
                    # Dynamo invalidated this entry (an object one of its
                    # guards holds by id died): it keeps its place in the
                    # list, with no code, and is never run again.
                    continue
                if (
                    ecode.co_freevars != freevars
                    or ecode.co_argcount != code.co_argcount
                    or ecode.co_kwonlyargcount != code.co_kwonlyargcount
                    or ecode.co_varnames[: nargs + code.co_kwonlyargcount]
                    != code.co_varnames[: nargs + code.co_kwonlyargcount]
                ):
                    state["disabled"] = True
                    logger.info(
                        "fast_compiled_call: %s disabled (transformed code "
                        "signature/closure differs)",
                        fn.__qualname__,
                    )
                    return
                gm = entry.guard_manager
                direct = types.FunctionType(
                    ecode, fn.__globals__, fn.__name__, fn.__defaults__, closure
                )
                direct.__kwdefaults__ = fn.__kwdefaults__
                if mru_idx < 0 and gm.check(f_locals):
                    mru_idx = len(built)
                built.append((gm, direct))
            # MRU entry must accept the frame the slow call just ran with.
            if mru_idx < 0:
                return
            if mru_idx > 0:
                built.insert(0, built.pop(mru_idx))
            state["entries"] = built
            state["count"] = _entry_count(code)
        except Exception:  # noqa: BLE001 - fall back to dynamo for good
            state["disabled"] = True
            logger.warning(
                "fast_compiled_call: %s disabled", fn.__qualname__, exc_info=True
            )

    def _note(fast: bool) -> None:
        _STATS[0 if fast else 1] += 1
        if (_STATS[0] + _STATS[1]) % (_LOG_EVERY // 4 if _MEMO_MODE else _LOG_EVERY) == 0:
            if _MEMO_MODE:
                logger.info(
                    "fast_compiled_call: fast=%d slow=%d t3_guard_memo(%s)=%s",
                    _STATS[0],
                    _STATS[1],
                    _MEMO_MODE,
                    _MEMO_STATS,
                )
            else:
                logger.info(
                    "fast_compiled_call: fast=%d slow=%d", _STATS[0], _STATS[1]
                )

    def _memo_dispatch(args: tuple, count: int) -> Any:
        """Memo lookup; returns the direct fn to call or None."""
        if state["memo_count"] != count:
            state["memo"] = {}
            state["memo_count"] = count
            return None
        key = _memo_key(args)
        if key is None:
            _MEMO_STATS["nokey"] += 1
            return None
        hit = state["memo"].get(key)
        if hit is None:
            return None
        if _MEMO_MODE == "check":
            f_locals = _f_locals(args, {})
            sel = None
            for gm, direct in state["entries"]:
                if gm.check(f_locals):
                    sel = direct
                    break
            # Guard sets can overlap; the memo is right whenever the memoized
            # entry's own guards accept the call.
            ok = sel is hit or any(
                d is hit and gm.check(f_locals) for gm, d in state["entries"]
            )
            if ok:
                _MEMO_STATS["check_ok"] += 1
            else:
                _MEMO_STATS["check_mismatch"] += 1
                if _MEMO_STATS["check_mismatch"] <= 20:
                    logger.warning(
                        "VLLM_TPU_T3_GUARD_MEMO=check: memo entry for %s fails "
                        "its guards (guard-selected: %s)",
                        fn.__qualname__,
                        "none" if sel is None else "other entry",
                    )
            if sel is None:
                return None
            hit = sel
        _MEMO_STATS["hit"] += 1
        return hit

    def _memo_store(args: tuple, direct: Any, count: int) -> None:
        if state["memo_count"] != count:
            state["memo"] = {}
            state["memo_count"] = count
        memo = state["memo"]
        if len(memo) >= _MEMO_MAX_ENTRIES:
            return
        key = _memo_key(args)
        if key is not None and key not in memo:
            memo[key] = direct
            _MEMO_STATS["store"] += 1

    @functools.wraps(fn)
    def fast_dispatch_call(*args: Any, **kwargs: Any) -> Any:
        # Inside a trace (an outer compiled function calling this one) dynamo
        # folds is_compiling() to True and inlines the compiled function,
        # without tracing the cache-entry and eval-frame queries below.
        if not _CONFIG["fast"] or torch.compiler.is_compiling():
            return compiled(*args, **kwargs)
        if not state["disabled"]:
            entries = state["entries"]
            if (
                entries
                and _c_eval_frame.get_eval_frame_callback() is None
                and stance_mod._stance.stance == "default"
            ):
                if _MEMO_MODE and not kwargs and len(args) == nargs:
                    count = _entry_count(code)
                    if count == state["count"]:
                        hit = _memo_dispatch(args, count)
                        if hit is not None:
                            _note(True)
                            return hit(*args)
                f_locals = _f_locals(args, kwargs)
                if f_locals is not None and _entry_count(code) == state["count"]:
                    for idx, (gm, direct) in enumerate(entries):
                        # A dead entry (invalidated, or freed by
                        # torch._dynamo.reset()) can leave the entry count
                        # unchanged, or a later compile can restore it, while
                        # its guards still pass: Dynamo sets the guard
                        # manager's cache_entry to None, so never run it.
                        if gm.cache_entry is None:
                            continue
                        if gm.check(f_locals):
                            if _MEMO_MODE and not kwargs and len(args) == nargs:
                                _memo_store(args, direct, state["count"])
                            if idx > 0:
                                entries.insert(0, entries.pop(idx))
                            if not state["logged"]:
                                state["logged"] = True
                                logger.info(
                                    "fast_compiled_call: first direct dispatch of %s",
                                    fn.__qualname__,
                                )
                            _note(True)
                            return direct(*args, **kwargs)
            else:
                f_locals = _f_locals(args, kwargs)
            out = compiled(*args, **kwargs)
            _note(False)
            _refresh(f_locals)
            return out
        return compiled(*args, **kwargs)

    fast_dispatch_call._fast_dispatch_state = state  # type: ignore[attr-defined]
    fast_dispatch_call._torchdynamo_orig_callable = getattr(  # type: ignore[attr-defined]
        compiled, "_torchdynamo_orig_callable", fn
    )
    # torch._dynamo.eval_frame.innermost_fn (used by torch._dynamo.disable)
    # follows _torchdynamo_orig_callable only from the wrapper it names.
    fast_dispatch_call._torchdynamo_wrapper_id = id(fast_dispatch_call)  # type: ignore[attr-defined]
    return fast_dispatch_call
