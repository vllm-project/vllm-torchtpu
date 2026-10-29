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

    __slots__ = ("samples", "template", "fill_args", "fill_sizes", "out_spec",
                 "ready", "dead", "reason")

    def __init__(self) -> None:
        self.samples: list[tuple[tuple, tuple, Any, Any]] = []
        self.template: list[Any] | None = None
        self.fill_args: list[tuple[int, int]] = []
        self.fill_sizes: list[tuple[int, int, int]] = []
        self.out_spec: Any = None
        self.ready = False
        self.dead = False
        self.reason = ""


def _describe(v: Any) -> str:
    if isinstance(v, torch.Tensor):
        kind = "Parameter" if isinstance(v, torch.nn.Parameter) else "Tensor"
        return f"{kind}{tuple(v.shape)}:{v.dtype}"
    return f"{type(v).__name__}={v!r}"[:80]


def _out_spec(ret: Any, out: Any) -> Any:
    """How the forward's return value is built from the graph outputs, or
    raise ValueError: ('o', k) for out[k], ('n',) for None, or a tuple/list
    of specs."""
    if ret is None:
        return ("n",)
    if isinstance(ret, (tuple, list)):
        return (type(ret), [_out_spec(r, out) for r in ret])
    for k, o in enumerate(out):
        if o is ret:
            return ("o", k)
    raise ValueError(f"returns {_describe(ret)}, not a graph output")


def _apply_spec(spec: Any, out: Any) -> Any:
    if spec[0] == "o":
        return out[spec[1]]
    if spec[0] == "n":
        return None
    return spec[0](_apply_spec(s, out) for s in spec[1])


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (tuple, list)):
        return (isinstance(b, type(a)) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    return a is b


def _classify(layout: _Layout) -> str | None:
    """Fill the layout from its recorded calls. Returns a reason when it can
    never be cached (layout.dead), None when ready or still undecided."""
    samples = layout.samples
    n = len(samples[0][1])
    if any(len(s[1]) != n for s in samples):
        return "flat length varies"
    kinds: list[tuple] = []
    for i in range(n):
        vals = [s[1][i] for s in samples]
        v0 = vals[0]
        argpos = [j for j, a in enumerate(samples[0][0]) if a is v0]
        if argpos and all(s[1][i] is s[0][argpos[0]] for s in samples):
            kinds.append((_ARG, argpos[0]))
            continue
        if isinstance(v0, int) and not isinstance(v0, bool):
            varied = len(set(vals)) >= 2
            # An int argument passed through.
            same_arg = [
                j for j, a in enumerate(samples[0][0])
                if isinstance(a, int) and not isinstance(a, bool)
                and all(isinstance(s[0][j], int) and s[0][j] == s[1][i]
                        for s in samples)
            ]
            if same_arg and varied:
                kinds.append((_ARG, same_arg[0]))
                continue
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
            if cands and varied:
                kinds.append((_SIZE, *min(cands)))
                continue
            if not varied:
                # Never varied yet: undecidable between a size and a constant.
                layout.reason = f"flat[{i}] int {v0} has not varied"
                return None
            return f"flat[{i}] int varies {sorted(set(vals))[:6]} with no matching arg size"
        if all(v is v0 for v in vals):
            if any(v0 is a for s in samples for a in s[0]):
                return f"flat[{i}] {_describe(v0)} is an argument only sometimes"
            kinds.append((_CONST, v0))
            continue
        same_shape = all(
            isinstance(v, torch.Tensor) and isinstance(v0, torch.Tensor)
            and v.shape == v0.shape for v in vals
        )
        return (f"flat[{i}] {_describe(v0)} changes identity across calls and is "
                f"not an argument (same shape: {same_shape}; "
                f"{len({id(v) for v in vals})} distinct of {len(vals)})")
    try:
        spec = _out_spec(samples[0][2], samples[0][3])
    except ValueError as e:
        return str(e)
    if not all(_same(_apply_spec(spec, s[3]), s[2]) for s in samples):
        return "return structure varies"
    layout.template = [k[1] if k[0] == _CONST else None for k in kinds]
    layout.fill_args = [(i, k[1]) for i, k in enumerate(kinds) if k[0] == _ARG]
    layout.fill_sizes = [(i, k[1], k[2]) for i, k in enumerate(kinds) if k[0] == _SIZE]
    layout.out_spec = spec
    layout.ready = True
    layout.samples = []
    return None


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
        self.undecided = 0
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
                reason = _classify(layout)
                if reason is not None:
                    layout.dead = True
                    layout.samples = []
                    logger.info("AOT_FLAT_ARGS %s: layout %s not cacheable: %s",
                                self.label, key[:2], reason)
                elif layout.ready:
                    logger.info(
                        "AOT_FLAT_ARGS %s: caching %d flat args (%d args, %d sizes)",
                        self.label, len(layout.template), len(layout.fill_args),
                        len(layout.fill_sizes))
                else:
                    layout.samples = layout.samples[-_RECORD_CALLS:]
                    if len(layout.samples) and self.calls == 0:
                        self.undecided += 1
                        if self.undecided in (1, 100, 1000):
                            logger.info("AOT_FLAT_ARGS %s: layout %s undecided: %s",
                                        self.label, key[:2], layout.reason)
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
        return _apply_spec(layout.out_spec, self.compiled(*self._build(layout, vals)))


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
