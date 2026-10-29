# SPDX-License-Identifier: Apache-2.0
"""Reuse the flat graph-argument list of an AOT-compiled forward across calls
(VLLM_TPU_AOT_FLAT_ARGS).

Dynamo's transformed forward (``AOTCompiledFunction.fn``) rebuilds the graph's
flat input list on every call: it walks ``self._modules[...]._parameters[...]``
attribute chains for every lifted parameter, buffer and tensor attribute (~700
for the Qwen3.8 backbone) before it calls the compiled graph, ~280 us of host
time per backbone launch. All of those resolve to the same objects on every
call; only the forward's own tensor arguments and the sizes taken from them
change.

So the first calls of each argument layout run the original forward with the
compiled graph wrapped by a recorder, and classify every flat position as
  * one of the forward's arguments (by identity),
  * a size of one of those arguments (an int that matched the same
    ``args[j].shape[d]`` on every recorded call, with at least two distinct
    values seen), or
  * a constant (the same object on every recorded call).
Once a layout is fully classified (and the forward returns graph outputs
as-is), later calls build the list from the template and call the compiled
graph directly. A forward whose generated code stores anything (attribute,
subscript, global) is left alone.

Safety: every VLLM_TPU_AOT_FLAT_ARGS_CANARY-th fast-path call instead runs the
original forward and compares its flat list with the template; a mismatch
logs an error and switches the cache off for that forward. With
VLLM_TPU_AOT_FLAT_ARGS_CHECK=N the first N eligible calls run the original
forward and count positions where the template would have differed (logged),
without using the fast path.
"""

from __future__ import annotations

import dis
from collections.abc import Callable
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_STORE_OPS = {
    "STORE_ATTR",
    "STORE_GLOBAL",
    "STORE_SUBSCR",
    "STORE_SLICE",
    "DELETE_ATTR",
    "DELETE_GLOBAL",
    "DELETE_SUBSCR",
}
_RECORD_CALLS = 16
_ARG, _SIZE, _CONST = 0, 1, 2


def _compiled_fn_name(fn: Callable) -> str | None:
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    names = [
        n
        for n in code.co_names
        if n.startswith("__compiled_fn") and callable(fn.__globals__.get(n))
    ]
    if len(names) != 1:
        return None
    if any(ins.opname in _STORE_OPS for ins in dis.get_instructions(code)):
        return None
    return names[0]


class _Layout:
    """What one argument layout of the forward needs to rebuild its flat list."""

    __slots__ = ("samples", "template", "fill_args", "fill_sizes", "out_index",
                 "out_tuple", "ready", "dead")

    def __init__(self) -> None:
        self.samples: list[tuple[tuple, tuple, Any, Any]] = []
        self.template: list[Any] | None = None
        self.fill_args: list[tuple[int, int]] = []
        self.fill_sizes: list[tuple[int, int, int]] = []
        self.out_index: int | None = None
        self.out_tuple: tuple[int, ...] | None = None
        self.ready = False
        self.dead = False


def _classify(layout: _Layout) -> None:
    """Fill the layout from its recorded calls, or mark it dead."""
    samples = layout.samples
    first_flat = samples[0][1]
    n = len(first_flat)
    if any(len(s[1]) != n for s in samples):
        layout.dead = True
        return
    kinds: list[tuple] = []
    for i in range(n):
        vals = [s[1][i] for s in samples]
        v0 = vals[0]
        argpos = [j for j, a in enumerate(samples[0][0]) if a is v0]
        if argpos and all(s[1][i] is s[0][argpos[0]] for s in samples):
            kinds.append((_ARG, argpos[0]))
            continue
        if isinstance(v0, int) and not isinstance(v0, bool):
            cands = None
            for args, flat, _, _ in samples:
                here = {
                    (j, d)
                    for j, a in enumerate(args)
                    if isinstance(a, torch.Tensor)
                    for d, size in enumerate(a.shape)
                    if size == flat[i]
                }
                cands = here if cands is None else cands & here
            if cands and len({s[1][i] for s in samples}) >= 2:
                kinds.append((_SIZE, *min(cands)))
                continue
            if all(v == v0 for v in vals):
                # Never varied: undecidable between a size and a constant.
                return
            layout.dead = True
            return
        if all(v is v0 for v in vals) and not any(
            v0 is a for s in samples for a in s[0]
        ):
            kinds.append((_CONST, v0))
            continue
        layout.dead = True
        return
    # The forward must return graph outputs unchanged.
    out_index = out_tuple = None
    ret0, out0 = samples[0][2], samples[0][3]
    if not isinstance(out0, (tuple, list)):
        layout.dead = True
        return
    idx = [k for k, o in enumerate(out0) if o is ret0]
    if idx and all(s[2] is s[3][idx[0]] for s in samples):
        out_index = idx[0]
    elif isinstance(ret0, tuple):
        pos = []
        for r in ret0:
            m = [k for k, o in enumerate(out0) if o is r]
            if not m:
                layout.dead = True
                return
            pos.append(m[0])
        if all(
            isinstance(s[2], tuple)
            and len(s[2]) == len(pos)
            and all(r is s[3][k] for r, k in zip(s[2], pos))
            for s in samples
        ):
            out_tuple = tuple(pos)
        else:
            layout.dead = True
            return
    else:
        layout.dead = True
        return
    layout.template = [k[1] if k[0] == _CONST else None for k in kinds]
    layout.fill_args = [(i, k[1]) for i, k in enumerate(kinds) if k[0] == _ARG]
    layout.fill_sizes = [(i, k[1], k[2]) for i, k in enumerate(kinds) if k[0] == _SIZE]
    layout.out_index, layout.out_tuple = out_index, out_tuple
    layout.ready = True
    layout.samples = []


class FlatArgsForward:
    """Drop-in for ``AOTCompiledFunction.fn`` that reuses the flat list."""

    def __init__(self, fn: Callable, name: str, label: str) -> None:
        self.fn = fn
        self.name = name
        self.label = label
        self.globals = fn.__globals__
        self.compiled = fn.__globals__[name]
        self.layouts: dict[tuple, _Layout] = {}
        self.off = False
        self.calls = 0
        self.checks = 0
        self.check_mismatches = 0
        self.canary = envs.VLLM_TPU_AOT_FLAT_ARGS_CANARY
        self.check = envs.VLLM_TPU_AOT_FLAT_ARGS_CHECK

    def _record(self, args: tuple, kwargs: dict) -> tuple[Any, tuple, Any]:
        seen: dict[str, Any] = {}
        compiled = self.compiled

        def recorder(*flat):
            seen["flat"] = flat
            seen["out"] = out = compiled(*flat)
            return out

        self.globals[self.name] = recorder
        try:
            ret = self.fn(*args, **kwargs)
        finally:
            self.globals[self.name] = compiled
        return ret, seen.get("flat"), seen.get("out")

    def _build(self, layout: _Layout, args: tuple) -> list[Any]:
        flat = list(layout.template)
        for i, j in layout.fill_args:
            flat[i] = args[j]
        for i, j, d in layout.fill_sizes:
            flat[i] = args[j].shape[d]
        return flat

    def _compare(self, layout: _Layout, args: tuple, flat: tuple) -> int:
        built = self._build(layout, args)
        if len(built) != len(flat):
            return max(len(built), len(flat))
        bad = 0
        for b, f in zip(built, flat):
            if isinstance(f, int) and not isinstance(f, bool):
                bad += b != f
            else:
                bad += b is not f
        return bad

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.off:
            return self.fn(*args, **kwargs)
        # Positional and keyword arguments as one tuple; the layout key pins
        # the keyword names and which arguments are None.
        vals = args + tuple(kwargs.values()) if kwargs else args
        key = (len(args), tuple(kwargs), tuple(
            None if a is None else type(a) for a in vals
        ))
        layout = self.layouts.get(key)
        if layout is None:
            layout = self.layouts[key] = _Layout()
        if layout.dead:
            return self.fn(*args, **kwargs)
        if not layout.ready:
            ret, flat, out = self._record(args, kwargs)
            if flat is None:
                layout.dead = True
                return ret
            layout.samples.append((vals, flat, ret, out))
            if len(layout.samples) >= _RECORD_CALLS:
                _classify(layout)
                if layout.dead:
                    logger.info("AOT_FLAT_ARGS %s: layout %s not cacheable",
                                self.label, key)
                elif layout.ready:
                    logger.info(
                        "AOT_FLAT_ARGS %s: caching %d flat args (%d args, %d sizes)",
                        self.label, len(layout.template), len(layout.fill_args),
                        len(layout.fill_sizes))
                else:
                    layout.samples = layout.samples[-_RECORD_CALLS:]
            return ret
        self.calls += 1
        if self.checks < self.check:
            ret, flat, _ = self._record(args, kwargs)
            self.checks += 1
            self.check_mismatches += self._compare(layout, vals, flat)
            if self.checks % 500 == 0 or self.checks == self.check:
                logger.info("AOT_FLAT_ARGS_CHECK %s checks=%d mismatched_positions=%d",
                            self.label, self.checks, self.check_mismatches)
            return ret
        if self.canary and self.calls % self.canary == 0:
            ret, flat, _ = self._record(args, kwargs)
            bad = self._compare(layout, vals, flat)
            if bad:
                self.off = True
                logger.error("AOT_FLAT_ARGS %s: canary found %d stale flat args; "
                             "cache off", self.label, bad)
            return ret
        out = self.compiled(*self._build(layout, vals))
        if layout.out_index is not None:
            return out[layout.out_index]
        return tuple(out[k] for k in layout.out_tuple)


def patch_aot_flat_args() -> None:
    if not envs.VLLM_TPU_AOT_FLAT_ARGS:
        return
    from torch._dynamo.aot_compile import AOTCompiledFunction

    if getattr(AOTCompiledFunction.__post_init__, "_tpu_flat_args", False):
        return
    original = AOTCompiledFunction.__post_init__

    def __post_init__(self) -> None:
        original(self)
        name = _compiled_fn_name(self.fn)
        label = getattr(getattr(self._artifacts, "original_code", None),
                        "co_qualname", "?")
        if name is None:
            logger.info("AOT_FLAT_ARGS %s: generated forward not cacheable", label)
            return
        self.fn = FlatArgsForward(self.fn, name, label)

    __post_init__._tpu_flat_args = True
    AOTCompiledFunction.__post_init__ = __post_init__
    logger.info("AOT-compiled forwards reuse their flat graph-argument list")
