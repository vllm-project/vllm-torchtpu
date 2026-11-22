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

* The guard memo (``runner.host_trims`` turns it on under the batch-1 rule;
  it only acts with ``fast_compiled_call``): on top of the direct dispatch,
  remember which cache entry the guard check selected for an argument
  *signature* (per tensor: shape, stride, dtype, device, requires_grad, its
  Python type, and which earlier tensor argument it is, if any; per Python
  scalar / None / str: its type and value; any other object: its id; lists and
  tuples element-wise) and dispatch later calls with the same signature to
  that entry without re-running ``guard_manager.check``. An entry keeps a weak
  reference to each object it keys by id (a strong one if the object takes no
  weak reference) and is used only while each is still that object, so a
  recycled id never matches. The memo is dropped whenever the code object's
  cache entries change (recompile or eviction). A guard on something the
  signature does not cover (a global, or an attribute of an argument object)
  is not re-checked on a memo hit, so a function whose compiled program
  depends on such state, and the state can change between calls with the same
  arguments, opts out with ``tpu_compile(guard_memo=False)``: every call of it
  runs the guard check. The DFlash and DSpark draft forwards do (their draft
  model reads its attention metadata from vLLM's forward context), and so does
  DFlash's context-KV update (its program depends on the block tables inside
  the metadata objects it takes as arguments).

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
import types
import weakref
from typing import Any

import torch
from torch._C._dynamo import eval_frame as _c_eval_frame

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# [fast, slow] calls, counted over every wrapped function of the process;
# logged at INFO every _LOG_EVERY calls so serving logs show the hit rate.
_STATS = [0, 0]
# The guard memo: off until runner.host_trims applies the batch-1 rule.
_MEMO = {"on": False}
_MEMO_STATS = {"hit": 0, "store": 0, "nokey": 0}
_MEMO_MAX_ENTRIES = 64
_MEMO_MAX_SEQ = 16
_SCALARS = (bool, int, float, str)


def _memo_key(args: tuple | list, objs: list | None = None) -> tuple | None:
    """Cheap argument signature for the guard memo, or None (not memoizable).

    Covers everything Dynamo's argument guards can see on these call sites for
    a fixed specialization: tensor metadata (not values: tensor values are
    never guarded) and which tensor arguments are the same tensor (Dynamo
    guards that distinct tensor inputs stay distinct and duplicated ones stay
    duplicated), scalar values, and object identity for anything else
    (``self`` of a compiled method, modules, owners). The objects keyed by
    identity are appended to ``objs``, in key order, so an entry can check
    later that each is still alive (see ``_memo_refs``)."""
    return _signature(args, {}, objs)


def _signature(args: tuple | list, tensor_ids: dict, objs: list | None) -> tuple | None:
    parts: list[Any] = []
    for a in args:
        if isinstance(a, torch.Tensor):
            alias = tensor_ids.setdefault(id(a), len(tensor_ids))
            # Dynamo's tensor guard checks the exact type too (a subclass or
            # an nn.Parameter gets its own cache entry).
            parts.append(
                (
                    a.shape,
                    a.stride(),
                    a.dtype,
                    a.device,
                    a.requires_grad,
                    type(a),
                    alias,
                )
            )
        elif a is None or type(a) in _SCALARS:
            parts.append((type(a), a))
        elif type(a) in (list, tuple):
            if len(a) > _MEMO_MAX_SEQ:
                return None
            sub = _signature(a, tensor_ids, objs)
            if sub is None:
                return None
            parts.append((type(a), sub))
        else:
            if objs is not None:
                objs.append(a)
            parts.append(("id", id(a)))
    return tuple(parts)


def _strong(obj: Any) -> Any:
    return lambda: obj


def _memo_refs(objs: list) -> tuple:
    """A reference to each object a memo key holds by id: weak where the
    object allows it, else strong (the entry then keeps it alive, so its id
    cannot be reused while the entry exists)."""
    refs = []
    for obj in objs:
        try:
            refs.append(weakref.ref(obj))
        except TypeError:
            refs.append(_strong(obj))
    return tuple(refs)


def _memo_live(refs: tuple, objs: list) -> bool:
    """Whether every object the entry keyed by id is still the one passed."""
    return all(ref() is obj for ref, obj in zip(refs, objs, strict=True))


def configure_guard_memo(enabled: bool) -> None:
    """Turn the guard memo on or off for this process (``runner.host_trims``)."""
    _MEMO["on"] = bool(enabled)


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


def wrap(compiled: Any, fn: types.FunctionType, guard_memo: bool = True) -> Any:
    """The fast-dispatch wrapper of ``compiled``, ``torch.compile``'s result
    for the plain function ``fn``; ``guard_memo=False`` keeps it out of the
    guard memo."""
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
        "memo": {},  # signature -> (direct fn, refs to its id-keyed objects)
        "memo_count": -1,
        "guard_memo": guard_memo,  # False: this function never uses the memo
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
        if (_STATS[0] + _STATS[1]) % _LOG_EVERY == 0:
            if _MEMO["on"]:
                logger.info(
                    "fast_compiled_call: fast=%d slow=%d guard_memo=%s",
                    _STATS[0],
                    _STATS[1],
                    _MEMO_STATS,
                )
            else:
                logger.info("fast_compiled_call: fast=%d slow=%d", _STATS[0], _STATS[1])

    def _memo_dispatch(args: tuple, count: int) -> Any:
        """Memo lookup; returns the direct fn to call or None."""
        if state["memo_count"] != count:
            state["memo"] = {}
            state["memo_count"] = count
            return None
        objs: list = []
        key = _memo_key(args, objs)
        if key is None:
            _MEMO_STATS["nokey"] += 1
            return None
        memo = state["memo"]
        hit = memo.get(key)
        if hit is None:
            return None
        direct, refs = hit
        if not _memo_live(refs, objs):
            # An object the entry was keyed by is gone and its id now names
            # another one: the guards must run again.
            del memo[key]
            return None
        _MEMO_STATS["hit"] += 1
        return direct

    def _memo_store(args: tuple, direct: Any, count: int) -> None:
        if state["memo_count"] != count:
            state["memo"] = {}
            state["memo_count"] = count
        memo = state["memo"]
        if len(memo) >= _MEMO_MAX_ENTRIES:
            # Drop the entries whose keyed objects are gone, then the oldest:
            # an entry holding an object that cannot be weakly referenced
            # never dies, and refusing to store would end the memo's hits.
            for dead in [
                k for k, (_, refs) in memo.items() if any(r() is None for r in refs)
            ]:
                del memo[dead]
            while len(memo) >= _MEMO_MAX_ENTRIES:
                del memo[next(iter(memo))]
        objs: list = []
        key = _memo_key(args, objs)
        if key is not None and key not in memo:
            memo[key] = (direct, _memo_refs(objs))
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
                use_memo = (
                    _MEMO["on"]
                    and state["guard_memo"]
                    and not kwargs
                    and len(args) == nargs
                )
                if use_memo:
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
                            if use_memo:
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
