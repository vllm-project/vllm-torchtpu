# SPDX-License-Identifier: Apache-2.0
"""Call a static-shape `torch.compile(backend="tpu")` region's compiled graph
directly once its inputs repeat (VLLM_TPU_DIRECT_DISPATCH).

Every call of a compiled region runs Dynamo's Python entry before it reaches
the graph the backend returned: the eval-frame wrapper, a config patch and
revert, the guard check, the transformed bytecode that loads each graph input
by its source, and the disable wrappers around the graph. For the small glue
regions of a decode step that entry costs more host time than the executable
call it leads to, and a steady decode step makes several of them.

`tpu_compile(fn)` compiles `fn` as `torch.compile(fn, backend="tpu",
fullgraph=True, dynamic=False)` does, through a backend that wraps the graph
the "tpu" backend returns. A call whose inputs match an earlier call's (the
structure of the arguments; each tensor's type, dtype, shape, stride, device
and requires_grad; which arguments are the same tensor; every other argument's
value, or identity for objects; grad and inference mode) calls that call's
graph directly: its inputs are read from the arguments through the sources
Dynamo recorded for the graph's placeholders, and the result is rebuilt as
that call's was. Any other call goes through Dynamo, which guards and
compiles as usual, and records the call.

A direct call does not re-check Dynamo's guards on anything the key leaves
out: state outside the arguments (globals, the attributes of an object
argument such as a module or a runner, which the key holds by identity),
tensor dispatch keys, autocast, default dtype and deterministic mode, and
storage overlap between distinct views of one buffer. It does not replay
Python side effects of the traced function either. Use it only for pure
tensor functions whose object arguments and globals do not change after
warmup (the key keeps each object argument alive).
"""

import enum
import functools
import inspect
import types
import weakref
from collections.abc import Callable
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Distinct inputs a region remembers. A region that sees more (an argument
# Dynamo does not specialize on, varying per call) leaves the rest on Dynamo.
_MAX_ENTRIES = 512

_VALUES = (int, float, bool, str, type(None), torch.dtype, torch.device, torch.Size)
# Dynamo source kinds the placeholders are read through, by class name and
# fields (torch 2.13 on the served image, 2.15 in the unit tests); any other
# kind keeps that input set on Dynamo. Sources that resolve to their base:
_PASSTHROUGH_SOURCES = frozenset(
    {
        "NNModuleSource",
        "UnspecializedNNModuleSource",
        "UnspecializedBuiltinNNModuleSource",
        "FSDPNNModuleSource",
        "NotNNModuleSource",
    }
)
# Sources that read `getattr(base, member)` / `base[index]`.
_ATTR_SOURCES = frozenset(
    {"AttrSource", "ParamBufferSource", "UnspecializedParamBufferSource"}
)
_ITEM_SOURCES = frozenset({"GetItemSource", "DictGetItemSource"})


def _tensor_part(x: torch.Tensor, ids: dict) -> tuple:
    return (
        type(x),
        x.dtype,
        x.shape,
        x.stride(),
        x.device,
        x.requires_grad,
        ids.setdefault(id(x), len(ids)),
    )


def _key_parts(x: Any, append: Callable, ids: dict) -> bool:
    """Appends `x`'s part of the dispatch key; False when `x` holds a value
    the key does not represent (such a call goes through Dynamo)."""
    t = type(x)
    if t is torch.Tensor or isinstance(x, torch.Tensor):
        append(_tensor_part(x, ids))
        return True
    if t is list or t is tuple:
        append((t, len(x)))
        for e in x:
            if type(e) is torch.Tensor:  # the common element, without recursing
                append(_tensor_part(e, ids))
            elif not _key_parts(e, append, ids):
                return False
        return True
    if t is dict:
        append((t, tuple(x)))
        return _key_parts(tuple(x.values()), append, ids)
    if t in _VALUES:
        append((t, x))
        return True
    if isinstance(x, enum.Enum) or t.__hash__ is object.__hash__:
        append(x)  # by identity; the key keeps it alive, so its id is not reused
        return True
    return False


def _dispatch_key(args: tuple, kwargs: dict) -> tuple | None:
    parts: list = [torch.is_grad_enabled(), torch.is_inference_mode_enabled()]
    ids: dict = {}
    if not _key_parts(args, parts.append, ids):
        return None
    if kwargs and not _key_parts(kwargs, parts.append, ids):
        return None
    return tuple(parts)


def _resolver(source: Any, locate: Callable[[str], Any], fn_globals: dict):
    """A function of (args, kwargs) that reads `source`, a Dynamo source of a
    graph placeholder; None for a source kind it does not read."""
    if hasattr(source, "local_name"):
        return locate(source.local_name)
    if type(source).__name__ == "GlobalSource":
        name = source.global_name
        return lambda a, k: fn_globals[name]
    base = getattr(source, "base", None)
    read = None if base is None else _resolver(base, locate, fn_globals)
    if read is None:
        return None
    kind = type(source).__name__
    if kind in _PASSTHROUGH_SOURCES:
        return read
    member = getattr(source, "member", None)
    if kind in _ATTR_SOURCES and isinstance(member, str):
        return lambda a, k: getattr(read(a, k), member)
    index = getattr(source, "index", None)
    if (
        kind in _ITEM_SOURCES
        and type(index) in (int, str)  # not e.g. a ConstDictKeySource
        and not getattr(source, "index_is_slice", False)
    ):
        return lambda a, k: read(a, k)[index]
    return None


def _locator(fn: Callable, args: tuple, kwargs: dict) -> Callable[[str], Any]:
    """Maps a parameter name of `fn` to a reader of that parameter for calls
    shaped like (args, kwargs): a positional index, a keyword, the `*args`
    tail, or the default."""
    sig = inspect.signature(fn, follow_wrapped=False)
    bound = sig.bind(*args, **kwargs).arguments
    readers: dict[str, Callable] = {}
    for i, (name, p) in enumerate(sig.parameters.items()):
        if p.kind is p.VAR_POSITIONAL:
            readers[name] = lambda a, k, i=i: a[i:]
        elif p.kind is p.VAR_KEYWORD:
            continue  # its sources stay unread: such a call stays on Dynamo
        elif name in kwargs:
            readers[name] = lambda a, k, name=name: k[name]
        elif name in bound and i < len(args):
            readers[name] = lambda a, k, i=i: a[i]
        elif name not in bound and p.default is not p.empty:
            readers[name] = lambda a, k, v=p.default: v
    return readers.get


class _Template:
    """How a call's result is rebuilt from its graph's outputs and inputs."""

    __slots__ = ("kind", "value", "children")

    def __init__(self, kind: str, value: Any = None, children: tuple = ()):
        self.kind, self.value, self.children = kind, value, children

    def build(self, outs, flat):
        if self.kind == "out":
            return outs[self.value]
        if self.kind == "in":
            return flat[self.value]
        if self.kind == "value":
            return self.value
        built = [c.build(outs, flat) for c in self.children]
        return built if self.kind == "list" else tuple(built)


def _template(ret: Any, outs, flat) -> _Template | None:
    if isinstance(ret, torch.Tensor):
        for kind, seq in (("out", outs), ("in", flat)):
            for i, x in enumerate(seq):
                if x is ret:
                    return _Template(kind, i)
        return None
    if type(ret) is list or type(ret) is tuple:
        children = tuple(_template(r, outs, flat) for r in ret)
        if any(c is None for c in children):
            return None
        return _Template("list" if type(ret) is list else "tuple", None, children)
    if type(ret) in _VALUES:
        return _Template("value", ret)
    return None


class _Entry:
    __slots__ = ("graph", "readers", "template")

    def __init__(self, graph, readers, template):
        self.graph, self.readers, self.template = graph, readers, template

    def __call__(self, args: tuple, kwargs: dict):
        flat = [read(args, kwargs) for read in self.readers]
        return self.template.build(self.graph(*flat), flat)


class _Graph:
    """The backend's graph for one Dynamo cache entry and the sources of its
    placeholders; while its region records a call, it keeps that call's
    graph inputs and outputs. Dynamo calls it with the graph's inputs as
    positional arguments."""

    def __init__(self, region: "DirectDispatch", graph: Callable, sources: list):
        self.region, self.graph, self.sources = region, graph, sources

    def __call__(self, *args):
        outs = self.graph(*args)
        rec = self.region._recording
        if rec is not None:
            rec.append((self, args, outs))
        return outs


class _Backend:
    """The "tpu" backend, wrapping each graph it returns in a `_Graph`.
    Dynamo reads optional attributes off a backend (`backend_ctx_ctor`,
    `compiler_name`, ...): they are the tpu backend's."""

    def __init__(self, region: "DirectDispatch"):
        from torch._dynamo import lookup_backend

        self._region = region
        self.tpu = lookup_backend("tpu")

    def __getattr__(self, name: str):
        if name in ("_region", "tpu"):  # not set yet (copy, unpickling)
            raise AttributeError(name)
        return getattr(self.tpu, name)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        sources = [
            getattr(n.meta.get("grapharg"), "source", None)
            for n in gm.graph.nodes
            if n.op == "placeholder"
        ]
        return _Graph(self._region, self.tpu(gm, example_inputs), sources)


class DirectDispatch:
    """`fn` compiled for the "tpu" backend, calling its graphs directly for
    inputs it has seen (module docstring)."""

    def __init__(self, fn: Callable):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self._backend = _Backend(self)
        self._compiled = torch.compile(
            fn, backend=self._backend, fullgraph=True, dynamic=False
        )
        self._entries: dict[tuple, _Entry | None] = {}
        self._recording: list | None = None
        self._notes: set[str] = set()
        _REGIONS.add(self)

    def __get__(self, obj, objtype=None):
        return self if obj is None else types.MethodType(self, obj)

    def __call__(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return self._compiled(*args, **kwargs)
        key = _dispatch_key(args, kwargs)
        entry = self._entries.get(key) if key is not None else None
        if entry is not None:
            return entry(args, kwargs)
        if key is None or key in self._entries or len(self._entries) >= _MAX_ENTRIES:
            return self._compiled(*args, **kwargs)
        self._recording = rec = []
        try:
            ret = self._compiled(*args, **kwargs)
        finally:
            self._recording = None
        entry, why = self._learn(rec, ret, args, kwargs)
        self._entries[key] = entry
        note = "direct calls on" if entry is not None else f"kept on Dynamo: {why}"
        if note not in self._notes:
            self._notes.add(note)
            logger.info("DIRECT_DISPATCH %s: %s", self.__qualname__, note)
        return ret

    def _learn(
        self, rec: list, ret: Any, args: tuple, kwargs: dict
    ) -> tuple[_Entry | None, str]:
        """The direct call of the graph the recorded call ran (called as
        Dynamo called it), or None and why: the call ran no single graph, or
        its inputs or result are not all reachable from the arguments."""
        if len(rec) != 1:
            return None, f"the call ran {len(rec)} graphs"
        graph, flat, outs = rec[0]
        if outs is None:
            outs = ()
        if not isinstance(outs, (list, tuple)):
            return None, f"its graph returned a {type(outs).__name__}"
        if getattr(self._backend.tpu, "backend_ctx_ctor", None) is not None:
            # Dynamo enters it around every call; a direct call would not
            return None, "the tpu backend has a backend_ctx_ctor"
        if len(graph.sources) != len(flat):
            return (
                None,
                f"{len(flat)} graph inputs for {len(graph.sources)} placeholders",
            )
        locate = _locator(self._fn, args, kwargs)
        readers = []
        for source, value in zip(graph.sources, flat):
            read = (
                None
                if source is None
                else _resolver(source, locate, self._fn.__globals__)
            )
            if read is None or read(args, kwargs) is not value:
                return None, f"graph input {source} is not read from the arguments"
            readers.append(read)
        template = _template(ret, outs, flat)
        if template is None:
            return (
                None,
                f"its result ({type(ret).__name__}) is not rebuilt from the graph",
            )
        return _Entry(graph.graph, readers, template), ""


_REGIONS: "weakref.WeakSet[DirectDispatch]" = weakref.WeakSet()


def reset() -> None:
    """Forgets every region's direct calls; call it with `torch._dynamo.reset()`,
    which drops the graphs they call."""
    for region in list(_REGIONS):
        region._entries.clear()


def tpu_compile(fn: Callable | None = None):
    """`torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)`, with
    direct graph calls under VLLM_TPU_DIRECT_DISPATCH; also a decorator."""
    if fn is None:
        return tpu_compile
    if envs.VLLM_TPU_DIRECT_DISPATCH:
        return DirectDispatch(fn)
    return torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
