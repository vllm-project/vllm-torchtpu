# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MK_WHOLE_DIRECT_EXEC: call a compiled program's executable directly.

A ``support_torch_compile`` call walks a chain of Python layers before the
executable runs: the decorator, ``AOTCompiledFunction.__call__``, the Dynamo
bytecode that gathers the graph operands, ``VllmSerializableFunction``, the
FX ``split_gm``, ``PiecewiseBackend``, the 1-tuple unwrap, the AOTAutograd
runtime wrapper (with its input-mutation epilogue) and
``TorchTpuCompiledExecutable.__call__``. At a few hundred operands the layers
after the bytecode cost tens of microseconds per call and do the same thing on
every call of one bucket.

``DirectExec`` keeps the bytecode, which re-reads every operand on every call,
and replaces the backend function the bytecode calls with an ``_Interceptor``.
The first call of each bucket goes down the normal chain while the interceptor
records what the chain resolves to: the one executable that ran and the
operands it was given, the ``copy_`` calls of the mutation epilogue, and where
each returned value came from, all matched by object identity. Later calls of
that bucket pick the operands out of the bytecode's argument list in the
recorded order, call the executable's runner, replay the epilogue copies and
rebuild the outputs. A call whose bucket was never recorded, or whose record is
not a plain "operands in, outputs out, copies back" shape, takes the normal
chain. With the flag off nothing here is constructed.
"""

from __future__ import annotations

import operator
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import mk_call_marks as _mkm

logger = init_logger(__name__)

_MARK_ATTR = "_dynamo_propagated_dynamic_indices"
# Tensor methods that change a tensor in place in a way the record cannot
# replay; seen during a recording, they make the bucket unsupported.
_UNREPLAYABLE_METHODS = ("set_", "as_strided_", "resize_", "detach_")
_CONSTANT_TYPES = (type(None), bool, int, float)

# Executable classes whose `__call__` is the end of the chain. Tests add a
# stand-in class; on TPU it is torch_tpu's TorchTpuCompiledExecutable.
LEAF_TYPES: list[type] = []


def _leaf_types() -> list[type]:
    if not LEAF_TYPES:
        try:
            from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
                TorchTpuCompiledExecutable,
            )
        except ImportError:
            return []
        LEAF_TYPES.append(TorchTpuCompiledExecutable)
    return LEAF_TYPES


class _Unsupported:
    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason


class _Entry:
    """One bucket's record: operand order, runner, epilogue copies, outputs."""

    __slots__ = (
        "runner",
        "boxed",
        "take",
        "n_in",
        "consts",
        "ints",
        "dups",
        "copies",
        "out_kind",
        "out_items",
        "marks",
        "flat",
    )

    def call(self, args: tuple) -> Any:
        # VLLM_TPU_HOST_PHASES: laps of the megakernel try_forward split.
        hp = _mkm.active
        for i, v in self.ints:
            if args[i] != v:
                return None
        for i, j in self.dups:
            if args[i] is not args[j]:
                return None
        if self.flat:
            # Unboxed, all tensors: itemgetter's tuple is the runner's argument.
            flat_ops = self.take(args)
            if hp:
                _mkm.lap("x_mkf_key")
                if _mkm.FLUSH:
                    _mkm.flush(flat_ops)
            outs = self.runner(flat_ops)
        else:
            if self.n_in == 1:
                ops = [self.take(args)]
            elif self.n_in:
                ops = list(self.take(args))
            else:
                ops = []
            for pos, value in self.consts:
                ops.insert(pos, value)
            if hp:
                _mkm.lap("x_mkf_key")
                if _mkm.FLUSH:
                    _mkm.flush(ops)
            outs = self.runner((ops,)) if self.boxed else self.runner(tuple(ops))
        if hp:
            _mkm.lap("x_mkf_runner")
        for ai, oi in self.copies:
            args[ai].copy_(outs[oi])
        if hp:
            _mkm.lap("x_mkf_epilogue")
        items = []
        for kind, idx in self.out_items:
            if kind == 0:
                items.append(outs[idx])
            elif kind == 1:
                items.append(args[idx])
            else:
                items.append(idx)
        for pos, dims in self.marks:
            t = items[pos]
            if hasattr(t, _MARK_ATTR):
                getattr(t, _MARK_ATTR).update(dims)
            else:
                setattr(t, _MARK_ATTR, set(dims))
        if self.out_kind is tuple:
            return tuple(items)
        if self.out_kind is list:
            return items
        return items[0]


class _Interceptor:
    """Stands in for the backend function the Dynamo bytecode calls."""

    def __init__(self, orig: Any, owner: DirectExec) -> None:
        self.orig = orig
        self.owner = owner
        self.entries: dict[tuple, Any] = {}
        self.k0: int | None = None
        self.int_idx: tuple[int, ...] = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.orig, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _mkm.active:
            _mkm.lap("x_mkf_gather")
        if kwargs or not args:
            return self.orig(*args, **kwargs)
        if self.k0 is None:
            self._layout(args)
        try:
            # What PiecewiseBackend picks the compiled bucket by: the SymInt
            # arguments, else the leading dimension of the first tensor.
            key = (len(args), args[self.k0].shape[0], *[args[i] for i in self.int_idx])
        except (AttributeError, IndexError):
            key = None
        if key is None or not all(type(args[i]) is int for i in self.int_idx):
            self.owner.stats["fallback"] += 1
            return self.orig(*args)
        entry = self.entries.get(key)
        if entry is None:
            return self._record(key, args)
        if type(entry) is _Entry:
            out = entry.call(args)
            if out is not None:
                self.owner.stats["direct"] += 1
                return out
        self.owner.stats["fallback"] += 1
        return self.orig(*args)

    def _layout(self, args: tuple) -> None:
        self.k0 = next(
            (i for i, a in enumerate(args) if isinstance(a, torch.Tensor)), len(args)
        )
        self.int_idx = tuple(i for i, a in enumerate(args) if type(a) is int)

    def _record(self, key: tuple, args: tuple) -> Any:
        stats = self.owner.stats
        stats["record"] += 1
        leaf_calls: list = []
        copies: list = []
        bad: list[str] = []
        versions = {
            i: a._version for i, a in enumerate(args) if isinstance(a, torch.Tensor)
        }
        grad_before = torch.is_grad_enabled()
        saved: list = []

        def leaf_call(orig_call):
            def call(exe, *a, **k):
                out = orig_call(exe, *a, **k)
                leaf_calls.append((exe, a, k, out))
                return out

            return call

        for cls in _leaf_types():
            saved.append((cls, "__call__", cls.__dict__.get("__call__")))
            cls.__call__ = leaf_call(cls.__call__)
        orig_copy = torch.Tensor.copy_

        def copy_(dst, src, *a, **k):
            copies.append((dst, src, bool(a or k), len(leaf_calls)))
            return orig_copy(dst, src, *a, **k)

        saved.append((torch.Tensor, "copy_", torch.Tensor.__dict__.get("copy_")))
        torch.Tensor.copy_ = copy_
        for name in _UNREPLAYABLE_METHODS:
            orig_m = getattr(torch.Tensor, name)

            def unreplayable(t, *a, _n=name, _o=orig_m, **k):
                bad.append(f"Tensor.{_n}")
                return _o(t, *a, **k)

            saved.append((torch.Tensor, name, torch.Tensor.__dict__.get(name)))
            setattr(torch.Tensor, name, unreplayable)
        try:
            result = self.orig(*args)
        finally:
            for cls, name, orig in reversed(saved):
                if orig is None:
                    delattr(cls, name)
                else:
                    setattr(cls, name, orig)
        if torch.is_grad_enabled() != grad_before:
            bad.append("grad mode changed")
        entry = self._build(args, leaf_calls, copies, versions, result, bad)
        self.entries[key] = entry
        if type(entry) is _Entry:
            logger.info(
                "direct exec %s: recorded bucket %s (%d operands, %d runner "
                "operands, %d epilogue copies, %d outputs)",
                self.owner.name,
                key[2:],
                len(args),
                entry.n_in + len(entry.consts),
                len(entry.copies),
                len(entry.out_items),
            )
        else:
            stats["unsupported"] += 1
            logger.warning(
                "direct exec %s: bucket %s takes the normal call path: %s",
                self.owner.name,
                key[2:],
                entry.reason,
            )
        return result

    @staticmethod
    def _build(args, leaf_calls, copies, versions, result, bad) -> Any:
        if bad:
            return _Unsupported(", ".join(sorted(set(bad))))
        if len(leaf_calls) != 1:
            return _Unsupported(f"{len(leaf_calls)} executable calls, need 1")
        exe, a, k, outs = leaf_calls[0]
        runner = getattr(exe, "_fast_runner", None)
        if runner is None or k.get("output_shapes"):
            return _Unsupported("executable has no fast runner for this call")
        if not isinstance(outs, (list, tuple)):
            return _Unsupported(f"runner returned {type(outs).__name__}")
        boxed = len(a) == 1 and type(a[0]) is list
        operands = a[0] if boxed else a
        arg_pos: dict[int, int] = {}
        dups = []
        for i, x in enumerate(args):
            if isinstance(x, torch.Tensor):
                j = arg_pos.setdefault(id(x), i)
                if j != i:
                    dups.append((i, j))
        out_pos: dict[int, int] = {}
        for i, x in enumerate(outs):
            if isinstance(x, torch.Tensor):
                out_pos.setdefault(id(x), i)
        take_idx = []
        consts = []
        for p, x in enumerate(operands):
            if isinstance(x, torch.Tensor):
                i = arg_pos.get(id(x))
                if i is None:
                    return _Unsupported(f"runner operand {p} is not a call argument")
                take_idx.append(i)
            elif type(x) in _CONSTANT_TYPES:
                consts.append((p, x))
            else:
                return _Unsupported(f"runner operand {p} is a {type(x).__name__}")
        copy_pairs = []
        for dst, src, extra, n_leaf in copies:
            ai = arg_pos.get(id(dst))
            oi = out_pos.get(id(src)) if isinstance(src, torch.Tensor) else None
            if extra or n_leaf != 1 or ai is None or oi is None:
                return _Unsupported("copy_ outside the runner-output epilogue")
            copy_pairs.append((ai, oi))
        copied = {ai for ai, _ in copy_pairs}
        fed = set(take_idx)
        for i, v in versions.items():
            if args[i]._version != v and i not in copied and i not in fed:
                return _Unsupported(f"argument {i} changed in place outside the runner")
        if isinstance(result, torch.Tensor):
            out_kind, items = None, [result]
        elif type(result) in (tuple, list):
            out_kind, items = type(result), list(result)
        else:
            return _Unsupported(f"result is a {type(result).__name__}")
        out_items = []
        marks = []
        for pos, x in enumerate(items):
            if isinstance(x, torch.Tensor):
                if id(x) in out_pos:
                    out_items.append((0, out_pos[id(x)]))
                elif id(x) in arg_pos:
                    out_items.append((1, arg_pos[id(x)]))
                else:
                    return _Unsupported(
                        f"output {pos} is neither a runner output nor an argument"
                    )
                extra_attrs = set(x.__dict__) - {_MARK_ATTR}
                if extra_attrs:
                    return _Unsupported(f"output {pos} carries {sorted(extra_attrs)}")
                if _MARK_ATTR in x.__dict__:
                    marks.append((pos, frozenset(getattr(x, _MARK_ATTR))))
            elif type(x) in _CONSTANT_TYPES:
                out_items.append((2, x))
            else:
                return _Unsupported(f"output {pos} is a {type(x).__name__}")
        e = _Entry()
        e.runner = runner
        e.boxed = boxed
        e.n_in = len(take_idx)
        e.take = operator.itemgetter(*take_idx) if take_idx else None
        e.consts = consts
        e.ints = [(i, x) for i, x in enumerate(args) if type(x) in (int, bool, float)]
        e.dups = dups
        e.copies = copy_pairs
        e.out_kind = out_kind
        e.out_items = out_items
        e.marks = marks
        e.flat = not boxed and not consts and len(take_idx) > 1
        return e


class DirectExec:
    """Calls a ``support_torch_compile`` module through its recorded executables.

    Called with the module's own arguments. Before the module has an AOT-compiled
    function, or while anything the decorator checks per call is set, it calls the
    module normally.
    """

    def __init__(self, program: Any, name: str) -> None:
        self.program = program
        self.name = name
        self.stats = {
            "direct": 0,
            "record": 0,
            "fallback": 0,
            "unsupported": 0,
            "module": 0,
        }
        self._fn_obj: Any = None
        self._fn: Any = None
        self._bid = ""
        self._skip_decorator = False
        self._interceptor: _Interceptor | None = None

    def _install(self, fn_obj: Any) -> None:
        self._fn_obj = fn_obj
        self._fn = fn_obj.fn
        g = fn_obj.fn.__globals__
        bid = self._bid = fn_obj._artifacts.backend_id
        orig = g[bid]
        if isinstance(orig, _Interceptor):
            orig = orig.orig
        self._interceptor = g[bid] = _Interceptor(orig, self)
        cc = self.program.vllm_config.compilation_config
        mode = getattr(cc, "cudagraph_mode", None)
        # The decorator enters maybe_use_cudagraph_partition_wrapper around the
        # call; it does something only in this configuration.
        partition = bool(
            mode is not None
            and mode.has_piecewise_cudagraphs()
            and getattr(cc, "use_inductor_graph_partition", False)
        )
        self._skip_decorator = not partition
        logger.info(
            "direct exec %s: installed on %s (decorator %s)",
            self.name,
            bid,
            "skipped" if self._skip_decorator else "kept",
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        prog = self.program
        fn_obj = prog.__dict__.get("aot_compiled_fn")
        if fn_obj is None or prog.do_not_compile:
            self.stats["module"] += 1
            return prog(*args, **kwargs)
        if fn_obj is not self._fn_obj:
            self._install(fn_obj)
        elif self._fn.__globals__.get(self._bid) is not self._interceptor:
            # The backend was re-bound: put the interceptor back, and drop its
            # records unless the backend is the one they were made through.
            g, ic = self._fn.__globals__, self._interceptor
            if g[self._bid] is not ic.orig:
                ic.orig, ic.entries, ic.k0 = g[self._bid], {}, None
            g[self._bid] = ic
        if (
            not self._skip_decorator
            or fn_obj._guard_check_enabled
            or torch.compiler.is_compiling()
            or _skip_compiled()
        ):
            self.stats["module"] += 1
            return prog(*args, **kwargs)
        if _mkm.active:
            _mkm.lap("x_mkf_de_check")
        return self._fn(prog, *args, **kwargs)


_FC: Any = None


def _skip_compiled() -> bool:
    """The decorator's per-call `skip_compiled` check."""
    global _FC
    if _FC is None:
        from vllm import forward_context

        _FC = forward_context
    return (
        _FC.is_forward_context_available() and _FC.get_forward_context().skip_compiled
    )
