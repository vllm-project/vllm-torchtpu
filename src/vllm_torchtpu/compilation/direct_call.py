# SPDX-License-Identifier: Apache-2.0
"""Small per-step compiled programs, called without Dynamo's per-call wrapper.

A ``torch.compile`` entry pays Dynamo's frame evaluation on every call (eval
frame swap, config patching, cache lookup, guard check) before it reaches the
compiled program: on a TPU host about 20 us of a 36-40 us call for a program
of a few ops, against 18 us for the same program called through the
bytecode Dynamo generated for it. A decode step calls several such programs.

``direct_compile`` compiles like ``torch.compile(fullgraph=True,
dynamic=False)``, captures the bytecode Dynamo generates for each
specialization (vLLM's ``TorchCompileWithNoGuardsWrapper`` does the same for
the model forward) and keys it by everything a ``dynamic=False`` trace of a
function of its arguments specializes on: each tensor's shape, strides,
dtype, device and ``requires_grad``, which arguments are the same tensor, the
values of plain Python arguments and the grad mode. A call with a known key
runs that bytecode directly; any other call goes through ``torch.compile``.

What the key does not cover, and a wrapped function must therefore not
depend on: state other than its arguments and module constants (attributes,
globals that change), global modes other than grad mode (autocast, default
dtype), and two different tensor arguments sharing storage. Mutating an
input in place is fine: the compiled callable writes it back.
"""

import functools
import types
from collections.abc import Callable
from typing import Any

import torch
from torch._dynamo.convert_frame import register_bytecode_hook

# Bytecode captured while a ``DirectCompiled`` call compiles: (old, new) code.
_captures: list[list[tuple[types.CodeType, types.CodeType]]] = []
_hook_installed = False


def _record_bytecode(old: types.CodeType, new: types.CodeType) -> None:
    if _captures:
        _captures[-1].append((old, new))


def _install_hook() -> None:
    global _hook_installed
    if not _hook_installed:
        register_bytecode_hook(_record_bytecode)
        _hook_installed = True


def _value_key(value: Any, tensor_ids: list[int]) -> Any:
    """Hashable specialization key of one argument, or raise TypeError."""
    if isinstance(value, torch.Tensor):
        alias = tensor_ids.index(id(value)) if id(value) in tensor_ids else -1
        tensor_ids.append(id(value))
        return (
            "T",
            tuple(value.shape),
            value.stride(),
            value.dtype,
            value.device,
            value.requires_grad,
            alias,
        )
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype)):
        return (type(value), value)
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_value_key(v, tensor_ids) for v in value))
    raise TypeError(type(value))


def _call_key(args: tuple[Any, ...]) -> tuple | None:
    tensor_ids: list[int] = []
    try:
        values = tuple(_value_key(a, tensor_ids) for a in args)
    except TypeError:
        return None
    return (torch.is_grad_enabled(), values)


class DirectCompiled:
    """``torch.compile(fn, fullgraph=True, dynamic=False, **kwargs)`` whose
    compiled specializations run without Dynamo's per-call wrapper."""

    def __init__(self, fn: Callable, **compile_kwargs: Any) -> None:
        _install_hook()
        self._fn = fn
        self._compiled = torch.compile(
            fn, fullgraph=True, dynamic=False, **compile_kwargs
        )
        self._direct: dict[tuple, Callable] = {}
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any) -> Any:
        key = _call_key(args)
        direct = self._direct.get(key) if key is not None else None
        if direct is not None:
            return direct(*args)
        _captures.append([])
        try:
            out = self._compiled(*args)
        finally:
            captured = _captures.pop()
        codes = [new for old, new in captured if old is self._fn.__code__]
        # Exactly one new specialization of this function was compiled for
        # this call: its bytecode serves every later call with the same key.
        # (A call that reuses an existing Dynamo entry compiles nothing and
        # keeps going through torch.compile.)
        if key is not None and len(codes) == 1:
            fn = self._fn
            self._direct[key] = types.FunctionType(
                codes[0], fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
            )
        return out


def direct_compile(fn: Callable | None = None, **compile_kwargs: Any) -> Any:
    """Decorator form of ``DirectCompiled``."""
    if fn is None:
        return functools.partial(direct_compile, **compile_kwargs)
    return DirectCompiled(fn, **compile_kwargs)
