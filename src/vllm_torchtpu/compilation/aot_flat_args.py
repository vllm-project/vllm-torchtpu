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
import re
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
_ARG, _SIZE, _CONST, _CTX = 0, 1, 2, 3
_CTX_WALK_NODES = 50000


def _forward_context() -> Any:
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
    except ImportError:
        return None
    return get_forward_context() if is_forward_context_available() else None


def _context_paths() -> dict[int, list[tuple]]:
    """Every tensor reachable from the active forward context (attributes,
    dict values, list items; no modules), by id, with its access paths."""
    root = _forward_context()
    found: dict[int, list[tuple]] = {}
    if root is None:
        return found
    stack: list[tuple[Any, tuple]] = [(root, ())]
    visited: set[int] = set()
    while stack and len(visited) < _CTX_WALK_NODES:
        obj, path = stack.pop()
        if isinstance(obj, torch.Tensor):
            found.setdefault(id(obj), []).append(path)
            for d, size in enumerate(obj.shape):
                found.setdefault(("dim", int(size)), []).append((path, d))
            continue
        if isinstance(obj, int) and not isinstance(obj, bool):
            found.setdefault(("int", obj), []).append(path)
            continue
        if id(obj) in visited or len(path) > 5:
            continue
        visited.add(id(obj))
        if isinstance(obj, dict):
            stack.extend((v, path + (("k", k),)) for k, v in obj.items()
                         if isinstance(k, (str, int)))
        elif isinstance(obj, (list, tuple)):
            stack.extend((v, path + (("i", i),)) for i, v in enumerate(obj))
        elif isinstance(obj, torch.nn.Module) or callable(obj):
            continue
        elif hasattr(obj, "__dict__"):
            stack.extend((v, path + (("a", k),)) for k, v in vars(obj).items())
        elif hasattr(obj, "__slots__"):
            for k in obj.__slots__:
                if hasattr(obj, k):
                    stack.append((getattr(obj, k), path + (("a", k),)))
    return found


def _module_tensor_names(module: Any) -> dict[int, list[str]]:
    """Every tensor attribute under a module tree (buffers, plain tensor
    attributes, tensors in list/tuple/dict attributes), by id."""
    names: dict[int, list[str]] = {}
    if not isinstance(module, torch.nn.Module):
        return names
    for prefix, mod in module.named_modules(remove_duplicate=False):
        for key, val in vars(mod).items():
            if key in ("_parameters", "_modules"):
                continue
            items = val.items() if isinstance(val, dict) else (
                enumerate(val) if isinstance(val, (list, tuple)) else [(None, val)])
            for sub, t in items:
                if isinstance(t, torch.Tensor) and not isinstance(t, torch.nn.Parameter):
                    name = f"{prefix}.{key}" if prefix else key
                    if key == "_buffers" or sub is not None:
                        name = f"{name}[{sub}]"
                    names.setdefault(id(t), []).append(name)
    return names


def _log_constants(label: str, module: Any, template: list[Any]) -> None:
    """Name the lifted constants that are neither parameters nor read through
    the forward context: the ones a reassignment (not an in-place update)
    would leave stale until the next canary."""
    names = _module_tensor_names(module)
    consts = [t for t in template
              if isinstance(t, torch.Tensor) and not isinstance(t, torch.nn.Parameter)]
    groups: dict[str, int] = {}
    unnamed = 0
    for t in consts:
        found = names.get(id(t))
        if not found:
            unnamed += 1
            continue
        pattern = re.sub(r"\.\d+\.", ".*.", found[0])
        pattern = re.sub(r"\[\d+\]", "[*]", pattern)
        key = f"{pattern} {list(t.shape)} {str(t.dtype).replace('torch.', '')}"
        groups[key] = groups.get(key, 0) + 1
    logger.info(
        "AOT_FLAT_ARGS %s: %d non-parameter constants (%d unnamed): %s",
        label, len(consts), unnamed,
        "; ".join(f"{n}x {k}" for k, n in sorted(groups.items(), key=lambda kv: -kv[1])),
    )


def _int_source(src: tuple, args: tuple) -> Any:
    kind = src[0]
    if kind == "arg":
        return args[src[1]]
    if kind == "size":
        return args[src[1]].shape[src[2]]
    if kind == "ctxsize":
        return _resolve(src[1]).shape[src[2]]
    return _resolve(src[1])


def _resolve(path: tuple) -> Any:
    obj = _forward_context()
    for kind, key in path:
        obj = getattr(obj, key) if kind == "a" else obj[key]
    return obj


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

    __slots__ = ("samples", "template", "fill_args", "fill_sizes", "fill_ctx", "out_spec",
                 "ready", "dead", "reason")

    def __init__(self) -> None:
        self.samples: list[tuple[tuple, tuple, Any, Any]] = []
        self.template: list[Any] | None = None
        self.fill_args: list[tuple[int, int]] = []
        self.fill_sizes: list[tuple[int, tuple, tuple]] = []
        self.fill_ctx: list[tuple[int, tuple, tuple]] = []
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
        argpos = [j for j, a in enumerate(samples[0][0])
                  if a is v0 and not isinstance(a, int)]
        if argpos and all(s[1][i] is s[0][argpos[0]] for s in samples):
            kinds.append((_ARG, argpos[0]))
            continue
        if isinstance(v0, int) and not isinstance(v0, bool):
            # Every int is filled from a source read on each call: an int
            # argument, a size of a tensor argument, an int field or a tensor
            # size in the forward context. Never a recorded constant.
            srcs = None
            for args, flat, _, _, ctxmap in samples:
                x = flat[i]
                here = {("arg", j) for j, a in enumerate(args)
                        if isinstance(a, int) and not isinstance(a, bool) and a == x}
                here |= {("size", j, d) for j, a in enumerate(args)
                         if isinstance(a, torch.Tensor)
                         for d, size in enumerate(a.shape) if size == x}
                here |= {("ctxint", path) for path in ctxmap.get(("int", x), ())}
                here |= {("ctxsize", path, d) for path, d in ctxmap.get(("dim", x), ())}
                srcs = here if srcs is None else srcs & here
            if not srcs:
                return (f"flat[{i}] int {sorted(set(vals))[:6]} has no argument or "
                        "forward-context source")
            order = {"arg": 0, "size": 1, "ctxsize": 2, "ctxint": 3}
            ranked = sorted(srcs, key=lambda c: (order[c[0]], str(c)))
            kinds.append((_SIZE, ranked[0], list(ranked[1:4])))
            continue
        # Read through the forward context: resolve it on every call, even
        # when it was the same object on every recorded one.
        ctx = None
        for s in samples:
            here = set(s[4].get(id(s[1][i]), ()))
            ctx = here if ctx is None else ctx & here
        if ctx:
            kinds.append((_CTX, min(ctx, key=len),
                          (v0.dtype, {tuple(v.shape) for v in vals})))
            continue
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
    layout.fill_ctx = [(i, k[1], k[2]) for i, k in enumerate(kinds) if k[0] == _CTX]
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
        self.refusals = 0
        self.canary = envs.VLLM_TPU_AOT_FLAT_ARGS_CANARY
        self.check = envs.VLLM_TPU_AOT_FLAT_ARGS_CHECK

    def _record(self, args: tuple, kwargs: dict) -> tuple[Any, tuple, Any]:
        seen: dict[str, Any] = {}
        compiled = self.compiled

        def recorder(*flat):
            seen["flat"] = flat
            seen["ctx"] = _context_paths()
            seen["out"] = out = compiled(*flat)
            return out

        self.globals[self.name] = recorder
        try:
            ret = self.fn(*args, **kwargs)
        finally:
            self.globals[self.name] = compiled
        self.last_ctx = seen.get("ctx", {})
        return ret, seen.get("flat"), seen.get("out")

    def _build(self, layout: _Layout, args: tuple, strict: bool = True) -> list[Any] | None:
        """The flat list for this call, or None when a per-call guard fails:
        an int's alternative sources disagree, or a forward-context tensor
        has another dtype or a shape not verified yet."""
        flat = list(layout.template)
        for i, j in layout.fill_args:
            flat[i] = args[j]
        for i, src, alts in layout.fill_sizes:
            v = _int_source(src, args)
            if strict:
                for alt in alts:
                    if _int_source(alt, args) != v:
                        return None
            flat[i] = v
        for i, path, (dtype, shapes) in layout.fill_ctx:
            t = _resolve(path)
            if t.dtype != dtype or (strict and tuple(t.shape) not in shapes):
                return None
            flat[i] = t
        return flat

    def _refused(self, layout: _Layout, args: tuple, kwargs: dict, vals: tuple) -> Any:
        """A guard refused the template: run the full walk, then learn from it.
        New shapes of forward-context tensors and int sources that agree with
        the walk are kept; a primary int source that disagrees kills the
        layout."""
        self.refusals += 1
        ret, flat, _ = self._record(args, kwargs)
        if flat is None or len(flat) != len(layout.template):
            layout.dead = True
            return ret
        for entry in layout.fill_sizes:
            i, src, alts = entry
            if _int_source(src, vals) != flat[i]:
                layout.dead = True
                logger.warning("AOT_FLAT_ARGS %s: flat[%d] int source %s disagrees "
                               "with the walk; layout off", self.label, i, src)
                return ret
            alts[:] = [a for a in alts if _int_source(a, vals) == flat[i]]
        for i, path, (dtype, shapes) in layout.fill_ctx:
            t = flat[i]
            if t is not _resolve(path) or t.dtype != dtype:
                layout.dead = True
                logger.warning("AOT_FLAT_ARGS %s: flat[%d] context path %s disagrees "
                               "with the walk; layout off", self.label, i, path)
                return ret
            shapes.add(tuple(t.shape))
        if self.refusals in (1, 10, 100, 1000) or self.refusals % 10000 == 0:
            logger.info("AOT_FLAT_ARGS %s: %d guard refusals (verified by the walk "
                        "and learned)", self.label, self.refusals)
        return ret

    def _compare(self, layout: _Layout, args: tuple, flat: tuple) -> int:
        built = self._build(layout, args, strict=False)
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
            layout.samples.append((vals, flat, ret, out, self.last_ctx))
            if len(layout.samples) >= _RECORD_CALLS:
                reason = _classify(layout)
                if reason is not None:
                    layout.dead = True
                    layout.samples = []
                    logger.info("AOT_FLAT_ARGS %s: layout %s not cacheable: %s",
                                self.label, key[:2], reason)
                elif layout.ready:
                    _log_constants(self.label, args[0] if args else None,
                                   layout.template)
                    logger.info(
                        "AOT_FLAT_ARGS %s: caching %d flat args (%d args, %d sizes, "
                        "%d forward-context fields)",
                        self.label, len(layout.template), len(layout.fill_args),
                        len(layout.fill_sizes), len(layout.fill_ctx))
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
            if self._build(layout, vals) is None:
                self.refusals += 1
            self.check_mismatches += self._compare(layout, vals, flat)
            if self.checks % 500 == 0 or self.checks == self.check:
                logger.info("AOT_FLAT_ARGS_CHECK %s checks=%d mismatched_positions=%d "
                            "guard_refusals=%d", self.label, self.checks,
                            self.check_mismatches, self.refusals)
            return ret
        if self.canary and self.calls % self.canary == 0:
            ret, flat, _ = self._record(args, kwargs)
            bad = self._compare(layout, vals, flat)
            if bad > 0:
                self.off = True
                logger.error("AOT_FLAT_ARGS %s: canary found %d stale flat args; "
                             "cache off", self.label, bad)
            return ret
        flat = self._build(layout, vals)
        if flat is None:
            return self._refused(layout, args, kwargs, vals)
        return _apply_spec(layout.out_spec, self.compiled(*flat))


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
