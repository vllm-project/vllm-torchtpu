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

:func:`tpu_compile` is the decorator for them: it compiles a function with
``torch.compile(backend="tpu", fullgraph=True, dynamic=False)`` and wraps the
result. The wrapper calls the compiled function and nothing else until the
worker arms ``fast_compiled_call`` from ``additional_config["tpu_compile"]``
(``compilation.compile_options``; off by default). It reads the option on
every call, so a function decorated at import, before the options are known,
is covered. Armed, it evaluates each cached entry's
``guard_manager.check(f_locals)`` in MRU order and invokes the transformed
Python code object directly on a guard hit, skipping Dynamo's per-call frame
evaluation; every guard Dynamo built is kept.

TODO: Remove this module once PyTorch has a public guarded call path for
``fullgraph`` compiled functions (https://github.com/pytorch/pytorch/issues/206560).
Until then it relies on Dynamo internals, checked against torch 2.13.0 (git
cf30153c4c13): ``torch._C._dynamo.eval_frame._get_cache_entries_for_region``
and ``_get_total_cache_entry_count``, each cache entry's ``code`` and
``guard_manager.check`` and ``cache_entry`` (None once Dynamo invalidates or
frees the entry), the compiled wrapper's ``_isolate_recompiles_id``,
``torch._dynamo.eval_frame._stance``, and calling an entry's transformed code
object directly. A torch upgrade must re-check them.
"""

from __future__ import annotations

import functools
import inspect
import types
from collections.abc import Callable
from typing import Any

import torch
from torch._C._dynamo import eval_frame as _c_eval_frame

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# [fast, slow] calls, counted over every wrapped function of the process;
# logged at INFO every _LOG_EVERY calls so serving logs show the hit rate.
_STATS = [0, 0]
_LOG_EVERY = 20000
# Armed by configure(); off until the worker applies tpu_compile.
_CONFIG = {"fast": False}


def configure(fast: bool) -> None:
    """Arm (or disarm) the direct dispatch of every :func:`tpu_compile`
    function of the process. The worker calls this before any compiled
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


def _wrap(compiled: Any, fn: types.FunctionType) -> Any:
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
        if (_STATS[0] + _STATS[1]) % _LOG_EVERY == 0:
            logger.info("fast_compiled_call: fast=%d slow=%d", _STATS[0], _STATS[1])

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


def tpu_compile[F: Callable[..., Any]](fn: F) -> F:
    """Compile ``fn`` for TPU with ``torch.compile(fn, backend="tpu",
    fullgraph=True, dynamic=False)`` and wrap the result for fast dispatch.
    Use it as a decorator, or call it on a function.

    The compile call is main's, so the compiled programs and their guards
    are those of the plain ``torch.compile`` call. Disarmed (the default) the
    wrapper calls the compiled function and nothing else. ``fast`` is read on
    every call, never at decoration.
    """
    if not isinstance(fn, types.FunctionType):
        raise TypeError(
            f"tpu_compile takes a plain function, got {type(fn).__name__}; "
            "compile a bound method or a module with torch.compile"
        )
    compiled = torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
    if compiled is fn:
        # Dynamo is disabled (TORCHDYNAMO_DISABLE=1): torch.compile hands back
        # the function itself, which has no cache entries to dispatch.
        return fn
    return _wrap(compiled, fn)
