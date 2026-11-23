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


def _rebuild(spec: tuple, outs: Any, args: Any) -> Any:
    """A nested output (kind 3 item) from the runner outputs and arguments."""
    typ, sub = spec
    items = []
    for kind, idx in sub:
        if kind == 0:
            items.append(outs[idx])
        elif kind == 1:
            items.append(args[idx])
        elif kind == 2:
            items.append(idx)
        else:
            items.append(_rebuild(idx, outs, args))
    return typ(items)


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
        "take_idx",
    )

    def call(self, args: tuple) -> Any:
        for i, v in self.ints:
            if args[i] != v:
                return None
        for i, j in self.dups:
            if args[i] is not args[j]:
                return None
        if self.flat:
            # Unboxed, all tensors: itemgetter's tuple is the runner's argument.
            outs = self.runner(self.take(args))
        else:
            if self.n_in == 1:
                ops = [self.take(args)]
            elif self.n_in:
                ops = list(self.take(args))
            else:
                ops = []
            for pos, value in self.consts:
                ops.insert(pos, value)
            outs = self.runner((ops,)) if self.boxed else self.runner(tuple(ops))
        for ai, oi in self.copies:
            args[ai].copy_(outs[oi])
        items = []
        for kind, idx in self.out_items:
            if kind == 0:
                items.append(outs[idx])
            elif kind == 1:
                items.append(args[idx])
            elif kind == 2:
                items.append(idx)
            else:
                items.append(_rebuild(idx, outs, args))
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


def _traced_call(fn: Any, args: Any, kwargs: Any = None, tensors: Any = None):
    """Run ``fn(*args, **kwargs)`` once, watching the executable calls, the
    ``Tensor.copy_`` calls and any in-place change no record can replay.
    ``tensors`` (default: the tensors of ``args``) are the ones whose version
    counters are compared. Returns (result, executable calls, copies, versions
    before, unreplayable)."""
    leaf_calls: list = []
    copies: list = []
    bad: list[str] = []
    if tensors is None:
        tensors = args
    versions = {}
    for i, a in enumerate(tensors):
        if isinstance(a, torch.Tensor) and not a.is_inference():
            versions[i] = a._version
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
        result = fn(*args, **(kwargs or {}))
    finally:
        for cls, name, orig in reversed(saved):
            if orig is None:
                delattr(cls, name)
            else:
                setattr(cls, name, orig)
    if torch.is_grad_enabled() != grad_before:
        bad.append("grad mode changed")
    return result, leaf_calls, copies, versions, bad


class _Interceptor:
    """Stands in for the backend function the Dynamo bytecode calls."""

    def __init__(self, orig: Any, owner: DirectExec) -> None:
        self.orig = orig
        self.owner = owner
        self.entries: dict[tuple, Any] = {}
        self.k0: int | None = None
        self.int_idx: tuple[int, ...] = ()
        # DirectExec snapshot building: (key, args) of the calls seen while set.
        self.capture: list | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.orig, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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
        if self.capture is not None:
            self.capture.append((key, args))
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
        result, leaf_calls, copies, versions, bad = _traced_call(self.orig, args)
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

        def item(x, pos, top):
            if isinstance(x, torch.Tensor):
                if id(x) in out_pos:
                    it = (0, out_pos[id(x)])
                elif id(x) in arg_pos:
                    it = (1, arg_pos[id(x)])
                else:
                    return f"output {pos} is neither a runner output nor an argument"
                extra_attrs = set(x.__dict__) - {_MARK_ATTR}
                if extra_attrs:
                    return f"output {pos} carries {sorted(extra_attrs)}"
                if _MARK_ATTR in x.__dict__:
                    if not top:
                        return f"output {pos} is a nested marked tensor"
                    marks.append((pos, frozenset(getattr(x, _MARK_ATTR))))
                return it
            if type(x) in _CONSTANT_TYPES:
                return (2, x)
            if type(x) in (tuple, list):
                # A nested tuple/list of outputs (e.g. the fold's record).
                sub = []
                for j, y in enumerate(x):
                    r = item(y, f"{pos}.{j}", False)
                    if type(r) is str:
                        return r
                    sub.append(r)
                return (3, (type(x), tuple(sub)))
            return f"output {pos} is a {type(x).__name__}"

        for pos, x in enumerate(items):
            r = item(x, pos, True)
            if type(r) is str:
                return _Unsupported(r)
            out_items.append(r)
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
        e.take_idx = tuple(take_idx)
        return e


class DirectExec:
    """Calls a ``support_torch_compile`` module through its recorded executables.

    Called with the module's own arguments. Before the module has an AOT-compiled
    function, or while anything the decorator checks per call is set, it calls the
    module normally.

    With ``snapshot`` (VLLM_TPU_DIRECT_EXEC_SNAPSHOT) a bucket whose runner
    operands are all either tensors of the call's own arguments or tensors held
    by the module tree also skips the Dynamo bytecode: see `_Snapshot`.
    """

    def __init__(self, program: Any, name: str, snapshot: bool = False) -> None:
        self.program = program
        self.name = name
        self.snapshot = snapshot
        self.stats = {
            "direct": 0,
            "record": 0,
            "fallback": 0,
            "unsupported": 0,
            "module": 0,
            "snapshot": 0,
            "snapshot_miss": 0,
        }
        self._fn_obj: Any = None
        self._fn: Any = None
        self._bid = ""
        self._skip_decorator = False
        self._interceptor: _Interceptor | None = None
        self._snaps: dict[tuple, Any] = {}
        self._snap_seen: dict[tuple, Any] = {}
        self._tree: _ModuleTree | None = None

    def _install(self, fn_obj: Any) -> None:
        self._fn_obj = fn_obj
        self._fn = fn_obj.fn
        self._snaps, self._snap_seen, self._tree = {}, {}, None
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
            cur = g[self._bid]
            if isinstance(cur, _Interceptor):  # another DirectExec's
                cur = cur.orig
            if cur is not ic.orig:
                ic.orig, ic.entries, ic.k0 = cur, {}, None
                self._snaps, self._snap_seen = {}, {}
            g[self._bid] = ic
        if (
            not self._skip_decorator
            or fn_obj._guard_check_enabled
            or torch.compiler.is_compiling()
            or _skip_compiled()
        ):
            self.stats["module"] += 1
            return prog(*args, **kwargs)
        if self.snapshot:
            return self._snapshot_call(prog, args, kwargs)
        return self._fn(prog, *args, **kwargs)

    def _snapshot_call(self, prog: Any, args: tuple, kwargs: dict) -> Any:
        try:
            sig, flat, _, _ = _flatten(args, kwargs)
            snap = self._snaps.get(sig)
        except TypeError:
            return self._fn(prog, *args, **kwargs)
        if type(snap) is _Snapshot:
            out = snap.call(flat)
            if out is not None:
                self.stats["snapshot"] += 1
                return out
            self.stats["snapshot_miss"] += 1
            return self._fn(prog, *args, **kwargs)
        if snap is not None:
            return self._fn(prog, *args, **kwargs)
        # Not built yet: watch this call's bytecode arguments.
        ic = self._interceptor
        assert ic is not None
        cap: list = []
        ic.capture = cap
        # One walk of the module tree per install, redone when any of its dicts
        # changed since (the walk would be stale).
        tree = self._tree
        if tree is None or tree.versions() != tree.walked:
            tree = self._tree = _ModuleTree(prog)
        before = tree.walked
        try:
            out = self._fn(prog, *args, **kwargs)
        finally:
            ic.capture = None
        self._observe(sig, flat, cap, tree, before)
        return out

    def _observe(self, sig, flat, cap, tree, before) -> None:
        """Two calls of one call signature, seen through the interceptor, make
        a snapshot when every runner operand resolves the same way in both."""
        ic = self._interceptor
        if len(cap) != 1:
            return self._snap_unsupported(sig, f"{len(cap)} backend calls")
        key, bc_args = cap[0]
        entry = ic.entries.get(key)
        if type(entry) is not _Entry:
            return self._snap_unsupported(sig, "the bucket has no direct record")
        if tree.versions() != before:
            return self._snap_unsupported(sig, "the call changed module state")
        built = _Snapshot.build(entry, bc_args, flat, tree, sig)
        if type(built) is _Unsupported:
            return self._snap_unsupported(sig, built.reason)
        prev = self._snap_seen.get(sig)
        if prev is None:
            self._snap_seen[sig] = built
            return None
        del self._snap_seen[sig]
        if prev.identity() != built.identity():
            return self._snap_unsupported(sig, "a held operand changed between calls")
        self._snaps[sig] = built
        logger.info(
            "direct exec %s: snapshot for bucket %s (%d runner operands: %d from "
            "the call, %d held; %d guarded dicts, %d guarded list slots)",
            self.name,
            key[2:],
            len(built.template),
            len(built.dyn),
            len(built.template) - len(built.dyn) - len(entry.consts),
            len(built.dicts),
            len(built.elems),
        )
        return None

    def _snap_unsupported(self, sig, reason: str) -> None:
        self._snaps[sig] = _Unsupported(reason)
        self._snap_seen.pop(sig, None)
        logger.warning(
            "direct exec %s: no snapshot for this call shape, the bytecode "
            "gathers the operands: %s",
            self.name,
            reason,
        )


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


# --------------------------------------------------------------------------
# Call signatures shared by the snapshot and by DirectCall.

_TENSOR = object()


def _flatten(args: tuple, kwargs: dict, unmark: bool = False):
    """(signature, tensors, args', kwargs') of one call.

    The signature holds every non-tensor value (it raises TypeError when one is
    unhashable), the structure of top-level lists and tuples, the shape of
    every tensor and the grad mode. The tensors are in argument order, list
    elements in place. With ``unmark``, top-level tensors carrying a Dynamo
    dynamic-dim mark are replaced by a view in args'/kwargs' and in the tensor
    list, exactly as ``_static_shape_args`` does (so the record can match the
    executable's operands by identity)."""
    sig: list = [torch.is_grad_enabled()]
    flat: list = []
    rargs = list(args) if unmark else args
    rkwargs = dict(kwargs) if unmark else kwargs

    def one(a, top):
        if isinstance(a, torch.Tensor):
            if unmark and top and getattr(a, "_dynamo_dynamic_indices", None):
                a = a[:]
            flat.append(a)
            sig.append(a.shape)
            return a
        if top and type(a) in (list, tuple):
            sig.append((type(a), len(a)))
            for x in a:
                one(x, False)
            return a
        if isinstance(a, (list, tuple, dict, set)):
            raise TypeError("nested container")
        sig.append(a)
        return a

    for i, a in enumerate(args):
        v = one(a, True)
        if unmark:
            rargs[i] = v
    for k, a in kwargs.items():
        sig.append(k)
        v = one(a, True)
        if unmark:
            rkwargs[k] = v
    t = tuple(sig)
    hash(t)
    return t, flat, rargs, rkwargs


# --------------------------------------------------------------------------
# VLLM_TPU_DIRECT_EXEC_SNAPSHOT: skip the bytecode too.


class _ModuleTree:
    """Where each tensor held by a module tree lives, and the containers that
    guard it: a tensor is reached through ``_modules`` / ``__dict__`` /
    ``_parameters`` / ``_buffers`` dicts (guarded by their version tag, which
    changes on any assignment) and list slots (guarded by identity). Plain
    objects and lists/tuples directly in a module's ``__dict__`` are walked one
    level deep."""

    def __init__(self, root: Any) -> None:
        from torch import nn

        self.held: dict[int, list] = {}  # id(tensor) -> [(kind, container, key)]
        self.parents: dict[int, list] = {}  # id(node) -> [(dict, parent node)]
        self.dicts: dict[int, dict] = {}
        nodes = [root]
        seen = {id(root)}

        def add(t, guard):
            self.held.setdefault(id(t), []).append(guard)

        def edge(child, d, parent):
            self.parents.setdefault(id(child), []).append((d, parent))
            self.dicts[id(d)] = d
            if id(child) not in seen:
                seen.add(id(child))
                nodes.append(child)

        while nodes:
            node = nodes.pop()
            if isinstance(node, nn.Module):
                for name in ("_parameters", "_buffers"):
                    d = node.__dict__.get(name) or {}
                    for k, t in d.items():
                        if isinstance(t, torch.Tensor):
                            add(t, ("d", d, node))
                for k, m in (node.__dict__.get("_modules") or {}).items():
                    if m is not None:
                        edge(m, node._modules, node)
                plain = node.__dict__
                deep = True
            else:
                plain = getattr(node, "__dict__", None) or {}
                deep = False
            for k, v in plain.items():
                if isinstance(v, torch.Tensor):
                    add(v, ("d", plain, node))
                elif isinstance(v, (list, tuple)):
                    for j, x in enumerate(v):
                        if isinstance(x, torch.Tensor):
                            add(x, ("l", v, j, plain, node))
                elif (
                    deep
                    and not isinstance(v, nn.Module)
                    and type(v).__module__.startswith("vllm_torchtpu")
                    and hasattr(v, "__dict__")
                ):
                    edge(v, plain, node)
        self.walked = self.versions()

    def versions(self) -> tuple:
        from torch._C._dynamo.guards import dict_version

        return tuple(dict_version(d) for d in self.dicts.values())

    def guards_for(self, t: torch.Tensor) -> tuple[list, list] | None:
        """The dicts and list slots whose change could replace `t` on its way
        from the root, over every path; None when `t` is not held."""
        places = self.held.get(id(t))
        if not places:
            return None
        dicts: dict[int, dict] = {}
        elems = []
        todo = []
        for p in places:
            if p[0] == "d":
                dicts[id(p[1])] = p[1]
                todo.append(p[2])
            else:
                _, lst, j, owner_dict, node = p
                if isinstance(lst, list):
                    elems.append((lst, j, t))
                dicts[id(owner_dict)] = owner_dict
                todo.append(node)
        seen = set()
        while todo:
            node = todo.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            for d, parent in self.parents.get(id(node), ()):
                dicts[id(d)] = d
                todo.append(parent)
        return list(dicts.values()), elems


class _Snapshot:
    """One call signature's runner operands without the bytecode: a template
    of the operands with the held tensors and constants in place, the slots
    filled from the call's own tensors, and the guards proving the held ones
    are still what the bytecode would read (dict version tags along every path
    from the module, identity of list slots)."""

    __slots__ = (
        "runner",
        "boxed",
        "flat",
        "template",
        "dyn",
        "dicts",
        "versions",
        "elems",
        "n_unique",
        "copies",
        "out_kind",
        "out_items",
        "marks",
        "held_ids",
    )

    @staticmethod
    def build(entry: _Entry, bc_args: tuple, flat: list, tree: _ModuleTree, sig):
        from torch._C._dynamo.guards import dict_version

        call_pos: dict[int, int] = {}
        for j, t in enumerate(flat):
            call_pos.setdefault(id(t), j)
        dicts: dict[int, dict] = {}
        elems: list = []
        held_ids = []

        def resolve(i):
            x = bc_args[i]
            j = call_pos.get(id(x))
            if j is not None:
                return ("c", j)
            g = tree.guards_for(x)
            if g is None:
                return None
            for d in g[0]:
                dicts[id(d)] = d
            elems.extend(g[1])
            held_ids.append(id(x))
            return ("h", x)

        # Integer operands of the bytecode must be fixed by the signature: the
        # dynamic dims are sizes of the call's tensors, which the signature holds.
        sizes = {d for t in flat for d in t.shape}
        for i, v in entry.ints:
            if type(v) is not int or v not in sizes:
                return _Unsupported(f"bytecode integer operand {i}={v} not a call size")
        n_ops = entry.n_in + len(entry.consts)
        const_at = dict(entry.consts)
        template: list = [None] * n_ops
        dyn = []
        it = iter(entry.take_idx)
        for p in range(n_ops):
            if p in const_at:
                template[p] = const_at[p]
                continue
            i = next(it)
            r = resolve(i)
            if r is None:
                x = bc_args[i]
                return _Unsupported(
                    f"operand {i} ({tuple(x.shape)} {x.dtype}) is neither a call "
                    "argument nor held by the module tree"
                )
            if r[0] == "c":
                dyn.append((p, r[1]))
            else:
                template[p] = r[1]
        for i, j in entry.dups:
            if bc_args[i] is not bc_args[j]:
                return _Unsupported("duplicate operands differ")
        copies = []
        for ai, oi in entry.copies:
            r = resolve(ai)
            if r is None:
                return _Unsupported(f"mutated operand {ai} is not resolvable")
            copies.append((r, oi))
        out_items = []
        for kind, idx in entry.out_items:
            if kind == 3:
                return _Unsupported("nested outputs")
            if kind == 1:
                r = resolve(idx)
                if r is None:
                    return _Unsupported(f"returned operand {idx} is not resolvable")
                out_items.append((1, r))
            else:
                out_items.append((kind, idx))
        sn = _Snapshot()
        sn.runner = entry.runner
        sn.boxed = entry.boxed
        sn.flat = not entry.boxed and not entry.consts
        sn.template = template
        sn.dyn = dyn
        sn.dicts = list(dicts.values())
        sn.versions = tuple(dict_version(d) for d in sn.dicts)
        sn.elems = elems
        sn.n_unique = len({id(t) for t in flat})
        sn.copies = copies
        sn.out_kind = entry.out_kind
        sn.out_items = out_items
        sn.marks = entry.marks
        sn.held_ids = tuple(held_ids)
        return sn

    def identity(self) -> tuple:
        return (self.held_ids, tuple(self.dyn), len(self.template))

    def call(self, flat: list) -> Any:
        from torch._C._dynamo.guards import dict_version

        if tuple(map(dict_version, self.dicts)) != self.versions:
            return None
        for lst, j, t in self.elems:
            if len(lst) <= j or lst[j] is not t:
                return None
        if len({id(t) for t in flat}) != self.n_unique:
            return None
        ops = self.template.copy()
        for p, j in self.dyn:
            ops[p] = flat[j]
        if self.flat:
            outs = self.runner(tuple(ops))
        elif self.boxed:
            outs = self.runner((ops,))
        else:
            outs = self.runner(tuple(ops))
        for (kind, src), oi in self.copies:
            (flat[src] if kind == "c" else src).copy_(outs[oi])
        items = []
        for kind, idx in self.out_items:
            if kind == 0:
                items.append(outs[idx])
            elif kind == 1:
                k, src = idx
                items.append(flat[src] if k == "c" else src)
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


# --------------------------------------------------------------------------
# Plain `torch.compile` functions (no vLLM decorator, no AOT artifact).


class _PlainCandidate:
    __slots__ = ("entry", "guards", "n_unique")


_MAX_CANDIDATES = 8
_MAX_TRIES = 3


class DirectCall:
    """Calls a plain ``torch.compile``'d function through its recorded executables.

    A plain compiled function pays the Dynamo frame evaluation with its full
    guard check, the AOTAutograd runtime wrapper and the executable's Python
    ``__call__`` on every call. ``DirectCall`` wraps the function the call site
    calls (``fn``, e.g. a ``_static_shape_args`` wrapper around the compiled
    function). Calls are keyed by their signature (`_flatten`: every
    non-tensor value, list structure, every tensor shape, grad mode). The first
    call of a signature (normally during warmup) goes the normal way while the
    executable, its operands, the mutation epilogue and the outputs are
    recorded by identity, as `_Interceptor` does; a call that compiled is not
    recorded (the next one is). Later calls of that signature check the
    record's tensor guards (Dynamo's `TensorGuards`: dtype, device, shape,
    stride, requires_grad, dispatch keys, in C++) and alias pattern, then call
    the runner on the call's own tensors. Anything else takes the normal call.

    What it does not re-check, like any guard-free path: globals the traced
    function read (module constants), and Dynamo's global-state guards beyond
    grad mode. Use it only for functions that read neither in a way that
    changes while serving.
    """

    def __init__(self, fn: Any, name: str, unmark: bool = True) -> None:
        self.fn = fn
        self.name = name
        self.unmark = unmark
        self.entries: dict[tuple, Any] = {}
        self.tries: dict[tuple, int] = {}
        self.stats = {"direct": 0, "record": 0, "fallback": 0, "unsupported": 0}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            sig, flat, _, _ = _flatten(args, kwargs)
            cands = self.entries.get(sig)
        except TypeError:
            self.stats["fallback"] += 1
            return self.fn(*args, **kwargs)
        if type(cands) is list:
            # TensorGuards fails on a repeated tensor: check the distinct ones.
            uniq = {id(t): t for t in flat}
            n_unique = len(uniq)
            gargs = flat if n_unique == len(flat) else list(uniq.values())
            for c in cands:
                if c.n_unique == n_unique and c.guards.check(*gargs):
                    out = c.entry.call(flat)
                    if out is not None:
                        self.stats["direct"] += 1
                        return out
        elif cands is not None:
            self.stats["fallback"] += 1
            return self.fn(*args, **kwargs)
        if self.tries.get(sig, 0) >= _MAX_TRIES or (
            cands is not None and len(cands) >= _MAX_CANDIDATES
        ):
            self.stats["fallback"] += 1
            return self.fn(*args, **kwargs)
        return self._record(sig, flat, args, kwargs, cands)

    def _record(self, sig, flat, args, kwargs, cands) -> Any:
        from torch._C._dynamo.guards import TensorGuards
        from torch._dynamo.utils import counters

        self.stats["record"] += 1
        if self.unmark:
            _, flat_r, rargs, rkwargs = _flatten(args, kwargs, unmark=True)
        else:
            flat_r, rargs, rkwargs = flat, args, kwargs
        graphs = counters["stats"]["unique_graphs"]
        result, leaf_calls, copies, versions, bad = _traced_call(
            self.fn, rargs, rkwargs, tensors=flat_r
        )
        if counters["stats"]["unique_graphs"] != graphs:
            # This call compiled; record the next one.
            self.tries[sig] = self.tries.get(sig, 0) + 1
            return result
        entry = _Interceptor._build(flat_r, leaf_calls, copies, versions, result, bad)
        if type(entry) is not _Entry:
            self.stats["unsupported"] += 1
            self.tries[sig] = _MAX_TRIES
            if cands is None:
                self.entries[sig] = entry
            logger.warning(
                "direct exec %s: call shape takes the normal call path: %s",
                self.name,
                entry.reason,
            )
            return result
        c = _PlainCandidate()
        c.entry = entry
        uniq = list({id(t): t for t in flat}.values())
        c.guards = TensorGuards(
            *uniq,
            dynamic_dims_sizes=[list(t.shape) for t in uniq],
            dynamic_dims_strides=[list(t.stride()) for t in uniq],
        )
        c.n_unique = len(uniq)
        if cands is None:
            cands = self.entries[sig] = []
        else:
            # A signature recorded before whose guards did not hold this time.
            self.tries[sig] = self.tries.get(sig, 0) + 1
        cands.append(c)
        logger.info(
            "direct exec %s: recorded a call shape (%d tensors, %d runner "
            "operands, %d epilogue copies, %d outputs)",
            self.name,
            len(flat),
            entry.n_in + len(entry.consts),
            len(entry.copies),
            len(entry.out_items),
        )
        return result


_DIRECT_CALLS: dict[int, DirectCall] = {}


def direct_call(fn: Any, name: str) -> DirectCall:
    """The process's one `DirectCall` for `fn` (records are shared by the
    warmup and the serving call sites)."""
    d = _DIRECT_CALLS.get(id(fn))
    if d is None or d.fn is not fn:
        d = _DIRECT_CALLS[id(fn)] = DirectCall(fn, name)
    return d
