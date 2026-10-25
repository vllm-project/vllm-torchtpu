"""Direct guard-checked cache-entry dispatch and GLOBAL_STATE guard filtering
for ``torch.compile(backend="tpu", fullgraph=True, dynamic=False)`` functions.

* ``QX_FAST_COMPILED_CALL=1``: wraps compiled plain functions to evaluate each
  cached entry's ``guard_manager.check(f_locals)`` in MRU order and invoke the
  transformed Python code object directly on a guard hit.
* ``QX_TPU_DROP_GLOBAL_STATE_GUARD=1``: filters out Dynamo ``GLOBAL_STATE``
  guards (CPU intra-op ``num_threads``, ``grad_mode``) so warmup entries are
  reused across threads while preserving all tensor and value guards.
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

_STATS = [0, 0]  # [fast, slow]
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


_AB_PERIOD_MS = float(os.environ.get("VLLM_TPU_T3_AB_PERIOD_MS", "0") or 0)


def _ab_on() -> bool:
    """VLLM_TPU_T3_AB_PERIOD_MS in-process A/B (see runner/t3_host.py)."""
    if _AB_PERIOD_MS <= 0:
        return True
    import time

    return int(time.time() * 1000.0 / _AB_PERIOD_MS) % 2 == 0


def memo_stats() -> dict[str, int]:
    return dict(_MEMO_STATS)
_LOG_EVERY = 20000
_INSTALLED = False


def _region_entries(code: types.CodeType, region: int) -> list:
    getter = getattr(_c_eval_frame, "_get_cache_entries_for_region", None)
    if getter is not None:
        return list(getter(code, region))
    return list(_c_eval_frame._debug_get_cache_entry_list(code))


def _entry_count(code: types.CodeType) -> int:
    return _c_eval_frame._get_total_cache_entry_count(code)


def _wrap(compiled: Any, fn: types.FunctionType) -> Any:
    code = fn.__code__
    region = getattr(compiled, "_isolate_recompiles_id", -1)
    flags = code.co_flags
    simple_sig = not (flags & inspect.CO_VARARGS or flags & inspect.CO_VARKEYWORDS)
    nargs = code.co_argcount
    arg_names = code.co_varnames[:nargs]
    freevars = code.co_freevars
    closure = fn.__closure__
    sig = inspect.signature(fn)
    stance_mod = torch._dynamo.eval_frame
    state: dict[str, Any] = {
        "entry": None,  # (guard_manager, direct_fn, entry_count)
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
        if freevars:
            for name, cell in zip(freevars, closure):
                f_locals[name] = cell.cell_contents
        return f_locals

    def _refresh(f_locals: dict | None) -> None:
        state["memo"] = {}
        state["entry"] = None
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
                if (
                    ecode.co_freevars != freevars
                    or ecode.co_argcount != code.co_argcount
                    or ecode.co_kwonlyargcount != code.co_kwonlyargcount
                    or ecode.co_varnames[: nargs + code.co_kwonlyargcount]
                    != code.co_varnames[: nargs + code.co_kwonlyargcount]
                ):
                    state["disabled"] = True
                    logger.info(
                        "QX_FAST_COMPILED_CALL: %s disabled (transformed code "
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
            if mru_idx < 0 or not built[mru_idx][0].check(f_locals):
                return
            if mru_idx > 0:
                built.insert(0, built.pop(mru_idx))
            count = _entry_count(code)
            state["entries"] = built
            state["count"] = count
            state["entry"] = (built[0][0], built[0][1], count)
        except Exception:  # noqa: BLE001 - fall back to dynamo for good
            state["disabled"] = True
            logger.warning(
                "QX_FAST_COMPILED_CALL: %s disabled", fn.__qualname__, exc_info=True
            )

    def _note(fast: bool) -> None:
        _STATS[0 if fast else 1] += 1
        if (_STATS[0] + _STATS[1]) % (_LOG_EVERY // 4 if _MEMO_MODE else _LOG_EVERY) == 0:
            if _MEMO_MODE:
                logger.info(
                    "QX_FAST_COMPILED_CALL: fast=%d slow=%d t3_guard_memo(%s)=%s",
                    _STATS[0],
                    _STATS[1],
                    _MEMO_MODE,
                    _MEMO_STATS,
                )
            else:
                logger.info(
                    "QX_FAST_COMPILED_CALL: fast=%d slow=%d", _STATS[0], _STATS[1]
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
    def qx_fast_compiled(*args: Any, **kwargs: Any) -> Any:
        if not state["disabled"]:
            entries = state["entries"]
            if (
                entries
                and _c_eval_frame.get_eval_frame_callback() is None
                and stance_mod._stance.stance == "default"
                and not torch.compiler.is_compiling()
            ):
                if _MEMO_MODE and not kwargs and len(args) == nargs and _ab_on():
                    count = _entry_count(code)
                    if count == state["count"]:
                        hit = _memo_dispatch(args, count)
                        if hit is not None:
                            _note(True)
                            return hit(*args)
                f_locals = _f_locals(args, kwargs)
                if f_locals is not None and _entry_count(code) == state["count"]:
                    for idx, (gm, direct) in enumerate(entries):
                        if gm.check(f_locals):
                            if _MEMO_MODE and not kwargs and len(args) == nargs:
                                _memo_store(args, direct, state["count"])
                            if idx > 0:
                                item = entries.pop(idx)
                                entries.insert(0, item)
                                state["entry"] = (
                                    item[0],
                                    item[1],
                                    state["count"],
                                )
                            if not state["logged"]:
                                state["logged"] = True
                                logger.info(
                                    "QX_FAST_COMPILED_CALL: first direct dispatch of %s",
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

    qx_fast_compiled._qx_fast_state = state  # type: ignore[attr-defined]
    qx_fast_compiled._torchdynamo_orig_callable = getattr(  # type: ignore[attr-defined]
        compiled, "_torchdynamo_orig_callable", fn
    )
    # torch._dynamo.eval_frame.innermost_fn (used by torch._dynamo.disable)
    # follows _torchdynamo_orig_callable only from the wrapper it names.
    qx_fast_compiled._torchdynamo_wrapper_id = id(qx_fast_compiled)  # type: ignore[attr-defined]
    return qx_fast_compiled


def _drop_global_state(entries: Any) -> list[bool]:
    return [e.guard_type != "GLOBAL_STATE" for e in entries]


def install(fast: bool = True, drop_global_state: bool = False) -> None:
    """Hook ``torch.compile`` for ``backend="tpu", fullgraph=True,
    dynamic=False`` plain functions. Idempotent.

    fast (QX_FAST_COMPILED_CALL=1): guard-checked direct dispatch (above).
    drop_global_state (QX_TPU_DROP_GLOBAL_STATE_GUARD=1): compile them with a
    ``guard_filter_fn`` that drops only the GLOBAL_STATE guard (grad mode,
    num_threads, autocast, default dtype, ...). Warmup entries are then
    reused by serving even when the serving thread's num_threads/grad mode
    differ from the warmup thread's; tensor guards (shape, stride, dtype,
    device, requires_grad) and value guards are kept. These graphs take no
    requires_grad inputs and run on TPU, so neither grad mode nor the CPU
    intra-op thread count changes what they compute.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    orig_compile = torch.compile

    def _eligible(model: Any, kwargs: dict) -> bool:
        return (
            isinstance(model, types.FunctionType)
            and kwargs.get("backend") == "tpu"
            and kwargs.get("fullgraph") is True
            and kwargs.get("dynamic") is False
            and not kwargs.get("disable", False)
            and kwargs.get("options") is None
            and kwargs.get("mode") is None
        )

    @functools.wraps(orig_compile)
    def qx_compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
        if args:
            return orig_compile(model, *args, **kwargs)
        if model is None:

            def deco(m: Any) -> Any:
                return qx_compile(m, **kwargs)

            return deco
        if not _eligible(model, kwargs):
            return orig_compile(model, **kwargs)
        ckw = dict(kwargs)
        if drop_global_state:
            # torch.compile pops guard_filter_fn; the emptied dict is falsy,
            # so the backend still gets no options kwarg.
            ckw["options"] = {"guard_filter_fn": _drop_global_state}
        out = orig_compile(model, **ckw)
        if fast and callable(out):
            return _wrap(out, model)
        return out

    torch.compile = qx_compile
    if fast:
        logger.info(
            "QX_FAST_COMPILED_CALL=1: torch.compile(backend='tpu', fullgraph=True, "
            "dynamic=False) functions dispatch guard-checked cache entries directly"
        )
    if drop_global_state:
        logger.info(
            "QX_TPU_DROP_GLOBAL_STATE_GUARD=1: torch.compile(backend='tpu', "
            "fullgraph=True, dynamic=False) functions compile without the "
            "GLOBAL_STATE guard (num_threads/grad_mode no longer force recompiles)"
        )
