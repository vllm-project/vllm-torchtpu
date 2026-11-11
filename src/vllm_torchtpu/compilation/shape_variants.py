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
"""Per-token-bucket specialization of the compiled backbone graph.

vLLM traces the model once with the token dimension dynamic and every guard
dropped (``TorchCompileWithNoGuardsWrapper``), then compiles that one FX graph
for every entry of ``compile_sizes`` (``PiecewiseBackend.compile_all_ranges``
feeds it concrete inputs per size via ``create_concrete_args``). A model that
decides anything from the token count *while tracing* -- a wave or chunk count,
a threshold branch -- freezes that decision at whichever bucket was traced
first, and every other bucket inherits it, so a bucket the graph was not traced
for fails in the shape it was frozen at.

Dynamo does record what a trace assumed -- the ShapeEnv keeps the shape guards
it discharged, even though vLLM drops them for runtime checking -- and it is
reachable from the FX graph the compiler adaptor is handed. So a graph is asked
to build only the buckets its guards accept, and a bucket it refuses is compiled
by a later trace and handed back through :class:`BucketExecutables`. The graph
that stays live therefore ends up with one executable per bucket.

Nothing here runs at serving time: dispatch stays vLLM's own
(``PiecewiseBackend.__call__`` picks ``range_entries[Range(n, n)].runnable``),
and each bucket is still compiled exactly once, so the number of TPU programs --
each holding an HBM program region for the life of the process -- is unchanged.
Only the Dynamo traces multiply, by one per structurally distinct group.

The partition follows from the guards alone, so every DP/TP rank derives the
same one and stays in lockstep through the collectives inside the specialized
region. The decision has to come from a tensor shape: that is what leaves a
guard behind. A branch on some other python value is guarded outside the
ShapeEnv, vLLM drops that guard, and nothing here can see it.
"""

from __future__ import annotations

import contextlib
import operator
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import sympy
import torch
import torch.fx as fx

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ShapeSpecializationError(RuntimeError):
    """A token bucket cannot be served by the graph asked to compile it."""


def bucket_gate_edge(min_tokens: int) -> int:
    """Return cap `c` so `T > c` selects power-of-2 buckets `>= min_tokens`."""
    m = int(min_tokens)
    if m <= 1:
        return 0
    return m // 2 if m & (m - 1) == 0 else m - 1


# ---------------------------------------------------------------------------
# What a trace assumed about the token count
# ---------------------------------------------------------------------------


def _placeholder_value(node: fx.Node) -> Any:
    return node.meta.get("example_value", node.meta.get("val"))


def _placeholder_nodes(graph: fx.GraphModule) -> list[fx.Node]:
    inner = getattr(graph, "graph", None)
    if inner is None:
        return []
    nodes: list[fx.Node] = []
    for node in inner.nodes:
        if node.op != "placeholder":
            break
        nodes.append(node)
    return nodes


def _placeholder_values(graph: fx.GraphModule) -> list[Any]:
    values = []
    for node in _placeholder_nodes(graph):
        value = _placeholder_value(node)
        if value is not None:
            values.append(value)
    return values


def trace_shape_env(graph: fx.GraphModule) -> Any | None:
    """The ShapeEnv of the trace this graph came from, if it has one."""
    for value in _placeholder_values(graph):
        mode = getattr(value, "fake_mode", None)
        if mode is not None and mode.shape_env is not None:
            return mode.shape_env
    return None


def _finite(value: Any) -> int | None:
    """A ValueRanges endpoint as an int, or None when it is infinite."""
    try:
        return int(value)
    except (AttributeError, TypeError):
        # sympy int_oo raises AttributeError and sympy.oo raises TypeError;
        # neither raises OverflowError.
        return None


def _holds_at(expr: sympy.Expr, substitution: dict) -> bool | None:
    """Whether a guard holds under the substitution; None if undecidable."""
    try:
        value = expr.subs(substitution)
        if not isinstance(value, sympy.logic.boolalg.BooleanAtom):
            value = sympy.simplify(value)
    except Exception:  # pragma: no cover - defensive; sympy is total here
        return None
    return bool(value) if isinstance(value, sympy.logic.boolalg.BooleanAtom) else None


def unsupported_reason(shape_env: Any, size: int) -> str | None:
    """Why a trace cannot serve ``size`` tokens, or None if it can.

    Every symbol of the trace takes the bucket size, which is the substitution
    vLLM itself performs to build the inputs this compilation will run on
    (``create_concrete_args``). Reading the trace's symbols rather than one
    subgraph's placeholders keeps the verdict identical for every subgraph of
    the same trace. An expression that stays symbolic is treated as satisfied:
    this only ever refuses a bucket it can prove wrong.
    """
    if shape_env is None:
        return None
    symbols = set(shape_env.var_to_range)
    if not symbols:
        return None

    for symbol in symbols:
        value_range = shape_env.var_to_range[symbol]
        lower, upper = _finite(value_range.lower), _finite(value_range.upper)
        # A backed size is guarded >= 2 only to dodge 0/1 specialization;
        # upstream discounts that same bound when it evaluates guards
        # (vllm/compilation/backends.py, evaluate_guards branch).
        if lower is not None and lower > 2 and size < lower:
            return f"{symbol} >= {lower}"
        if upper is not None and size > upper:
            return f"{symbol} <= {upper}"

    substitution = {symbol: size for symbol in symbols}
    for guard in shape_env.guards:
        expr = guard.expr
        if _holds_at(expr, substitution) is False:
            return str(expr)
    return None


def _meta_entry(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return type(value).__name__
    dims = tuple("?" if isinstance(d, torch.SymInt) else int(d) for d in value.shape)
    return (str(value.dtype), dims)


def _signature(graph: fx.GraphModule) -> tuple:
    return tuple(_meta_entry(value) for value in _placeholder_values(graph))


def _handover_inputs(graph: fx.GraphModule) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (node.name, _meta_entry(_placeholder_value(node)))
        for node in _placeholder_nodes(graph)
    )


def _input_permutation(
    stored: Sequence[tuple[str, Any]], current: Sequence[tuple[str, Any]]
) -> list[int] | None:
    if len(stored) != len(current):
        return None
    position = {entry: i for i, entry in enumerate(current)}
    perm: list[int] = []
    for entry in stored:
        idx = position.get(entry)
        if idx is None:
            return None
        perm.append(idx)
    return perm


def _scalar_values(graph: fx.GraphModule) -> tuple[Any, ...]:
    out: list[Any] = []
    for node in _placeholder_nodes(graph):
        value = _placeholder_value(node)
        if isinstance(value, torch.SymInt):
            out.append(value.node.expr)
        elif isinstance(value, int):
            out.append(value)
        else:
            out.append(None)
    return tuple(out)


def _concretize(expr: Any, size: int) -> int:
    if isinstance(expr, int) and not isinstance(expr, bool):
        return expr
    sym = sympy.sympify(expr)
    if len(sym.free_symbols) > 1:
        raise ShapeSpecializationError(
            f"scalar placeholder {expr} depends on multiple symbols: {sym.free_symbols}"
        )
    val = sym.subs({symbol: size for symbol in sym.free_symbols})
    if not getattr(val, "is_integer", False):
        raise ShapeSpecializationError(
            f"scalar placeholder {expr} at size={size} is non-integer: {val}"
        )
    return int(val)


def _scalar_adapter(
    stored: Sequence[tuple[str, Any]],
    stored_scalars: Sequence[Any],
    current: Sequence[tuple[str, Any]],
    current_scalars: Sequence[Any],
    size: int | None,
) -> tuple[Callable[[tuple], tuple], list[str]] | None:
    if size is None:
        return None
    stored_t = [i for i, v in enumerate(stored_scalars) if v is None]
    current_t = [i for i, v in enumerate(current_scalars) if v is None]
    if len(stored_t) != len(current_t):
        return None
    curr_names = [current[j][0] for j in current_t]
    curr_by_name = (
        {current[j][0]: j for j in current_t}
        if len(set(curr_names)) == len(curr_names)
        else {}
    )
    source: list[int] = [-1] * len(stored_t)
    for k, i in enumerate(stored_t):
        sname, smeta = stored[i]
        j = curr_by_name.get(sname)
        if j is not None:
            if current[j][1] != smeta:
                return None
            source[k] = j
    matched_j = [j for j in source if j != -1]
    if len(set(matched_j)) != len(matched_j):
        return None
    unmatched_k = [k for k, j in enumerate(source) if j == -1]
    if unmatched_k:
        used = set(matched_j)
        rem_j = [j for j in current_t if j not in used]
        if any(
            stored[stored_t[k]][1] != current[j][1]
            for k, j in zip(unmatched_k, rem_j, strict=True)
        ):
            return None
        if len(unmatched_k) > 1 and len(
            {stored[stored_t[k]][1] for k in unmatched_k}
        ) != len(unmatched_k):
            return None
        for k, j in zip(unmatched_k, rem_j, strict=True):
            source[k] = j
    plan: list[tuple[bool, Any]] = [(False, None)] * len(stored)
    for i, j in zip(stored_t, source, strict=True):
        plan[i] = (True, j)
    filled: list[str] = []
    for i, value in enumerate(stored_scalars):
        if value is not None:
            concrete_val = _concretize(value, size)
            plan[i] = (False, concrete_val)
            filled.append(f"{stored[i][0]}={value}->{concrete_val}")

    def adapt(args: tuple) -> tuple:
        return tuple(args[x] if is_arg else x for is_arg, x in plan)

    return adapt, filled


def _signature_delta(stored: Sequence[Any], current: Sequence[Any]) -> str:
    def render(counts: Counter) -> str:
        if not counts:
            return "nothing"
        by_count = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        return ", ".join(f"{count}x {item}" for item, count in by_count)

    return (
        f"{len(stored)} inputs vs {len(current)}. "
        f"Only in earlier trace: {render(Counter(stored) - Counter(current))}. "
        f"Only in current trace: {render(Counter(current) - Counter(stored))}"
    )


def _non_tensor_inputs(graph: fx.GraphModule) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in _placeholder_nodes(graph):
        value = _placeholder_value(node)
        if value is not None and not isinstance(value, torch.Tensor):
            out[node.name] = str(value)
    return out


def _named_delta(
    stored: Sequence[tuple[str, Any]],
    current: Sequence[tuple[str, Any]],
    stored_scalars: dict[str, str],
    current_scalars: dict[str, str],
) -> str:
    def render(entries: list[tuple[str, Any]], scalars: dict[str, str]) -> str:
        if not entries:
            return "nothing"
        shown = [
            f"{name}={scalars[name]}" if name in scalars else f"{name}:{meta}"
            for name, meta in entries[:8]
        ]
        more = f" (+{len(entries) - 8} more)" if len(entries) > 8 else ""
        return ", ".join(shown) + more

    only_stored = [e for e in stored if e not in set(current)]
    only_current = [e for e in current if e not in set(stored)]
    return (
        f"by name, only in earlier trace: {render(only_stored, stored_scalars)}; "
        f"only in current trace: {render(only_current, current_scalars)}"
    )


def _handover_error(key: str, delta: str) -> ShapeSpecializationError:
    return ShapeSpecializationError(
        f"{key} was compiled by another trace of this model whose inputs "
        f"differ beyond reordering or scalar SymInts ({delta}). Both sides of "
        "a token-count branch must lift the same tensor buffers and subgraph "
        "boundaries for bucket hand-over; unset TPU_KERNEL_ITER_MODE if set."
    )


def graph_signature(graph: fx.GraphModule) -> tuple:
    """:func:`_signature`, for callers outside this module.

    Stable across processes -- dtypes as strings, dims as ints or "?" -- so a
    compile cache written by one run can check that the graph it is being
    replayed against still takes the same inputs.
    """
    return _signature(graph)


# ---------------------------------------------------------------------------
# Handing a bucket from the trace that cannot serve it to the one that can
# ---------------------------------------------------------------------------


class BucketExecutables:
    """What earlier traces of the model being warmed up already compiled.

    Keyed by the compile cache directory -- one per compiled module and rank --
    and vLLM's own per-(subgraph, bucket) artifact key. Compile-time only:
    :func:`warmup` creates it and drops it, and nothing reads it while serving.
    Handing the object over rather than reloading the pickled artifact keeps one
    executable, and so one HBM program region, per bucket.
    """

    def __init__(self) -> None:
        self._runnables: dict[
            tuple[str | None, str],
            tuple[Callable[..., Any], tuple, tuple, dict[str, str], tuple],
        ] = {}
        self.refused: set[int] = set()
        self.traces = 0

    def get(
        self,
        cache_dir: str | None,
        key: str,
        graph: fx.GraphModule,
        size: int | None = None,
    ) -> Callable[..., Any] | None:
        found = self._runnables.get((cache_dir, key))
        if found is None:
            return None
        runnable, stored, stored_signature, stored_named, stored_scalars = found
        current = _handover_inputs(graph)
        if stored == current:
            return runnable
        perm = None
        if sorted(name for name, _ in stored) != sorted(name for name, _ in current):
            signature = _signature(graph)
            if signature == stored_signature:
                return runnable
            delta = _signature_delta(stored_signature, signature)
        else:
            perm = _input_permutation(stored, current)
            delta = _signature_delta(stored, current)
        if perm is None:
            adapter = _scalar_adapter(
                stored, stored_scalars, current, _scalar_values(graph), size
            )
            if adapter is None:
                raise _handover_error(
                    key,
                    delta
                    + "; "
                    + _named_delta(
                        stored, current, stored_named, _non_tensor_inputs(graph)
                    ),
                )
            adapt, filled = adapter
            logger.info(
                "[shape-variants] %s handed over across scalar-input difference "
                "(%d inputs vs %d; bucket %s scalars: %s)",
                key,
                len(stored),
                len(current),
                size,
                ", ".join(filled) or "none",
            )

            def adapted(*args: Any) -> Any:
                return runnable(*adapt(args))

            return adapted
        logger.info(
            "[shape-variants] %s handed over with %d of %d inputs reordered",
            key,
            sum(j != i for i, j in enumerate(perm)),
            len(perm),
        )
        to_stored_order = operator.itemgetter(*perm)

        def reordered(*args: Any) -> Any:
            return runnable(*to_stored_order(args))

        return reordered

    def put(
        self,
        cache_dir: str | None,
        key: str,
        graph: fx.GraphModule,
        runnable: Callable[..., Any],
    ) -> None:
        self._runnables[(cache_dir, key)] = (
            runnable,
            _handover_inputs(graph),
            _signature(graph),
            _non_tensor_inputs(graph),
            _scalar_values(graph),
        )

    def needs_retrace(self, size: int) -> bool:
        """Whether a graph refused this bucket and no trace has built it yet.

        Built sizes cannot answer this: every module compiled inside the
        window shares one window, so a bucket some other module built would
        mask this one's refusal. A refusal is recorded only once the hand-over
        lookup has missed, and :func:`retrace` clears the set, so what is left
        is exactly what the traces made so far cannot serve.
        """
        return size in self.refused


_session: BucketExecutables | None = None


@contextlib.contextmanager
def warmup() -> Iterator[BucketExecutables]:
    """Scope in which traces of one model may hand buckets to one another."""
    global _session
    assert _session is None, "nested shape-specialization warmups"
    _session = BucketExecutables()
    try:
        yield _session
    finally:
        _session = None


def in_warmup() -> bool:
    return _session is not None


def runnable_for(
    cache_dir: str | None, key: str, graph: fx.GraphModule, size: int | None = None
) -> Callable[..., Any] | None:
    """Return a handed-over or refusing runnable, or None to compile normally."""
    ready = _session.get(cache_dir, key, graph, size) if _session else None
    if ready is not None:
        logger.info("[shape-variants] %s handed over by an earlier trace", key)
        return ready

    reason = None if size is None else unsupported_reason(trace_shape_env(graph), size)
    if reason is None:
        return None
    if _session is None:
        raise ShapeSpecializationError(
            f"Token bucket {size} cannot be served by the FX graph vLLM traced "
            f"for this model: it assumes {reason}. The model branches on the "
            "token count while tracing, so every compile_size would share the "
            "structure of the one that was traced. Only the backbone is "
            "re-traced per bucket group today (see _precompile_backbone); make "
            "this module's structure independent of the token count."
        )

    logger.info("[shape-variants] %s left to a later trace: %s", key, reason)
    assert size is not None
    _session.refused.add(size)

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ShapeSpecializationError(
            f"{key} was left to a later trace ({reason}) and never compiled; "
            "the warmup ladder is out of step with the partition."
        )

    return refuse


def remember(
    cache_dir: str | None, key: str, graph: fx.GraphModule, runnable: Callable[..., Any]
) -> None:
    """Offer a bucket, freshly compiled or loaded, to the traces that follow."""
    if _session is not None:
        _session.put(cache_dir, key, graph, runnable)


def aot_cache_tag() -> str | None:
    """Discriminator for vLLM's AOT artifact path while several traces run.

    The path is keyed by the config and the forward code, which do not tell two
    traces of one model apart, so without this the second trace loads the
    first's artifact instead of tracing (and every bucket gets the first
    structure again). The trace ordinal is enough: the partition is a function
    of the guards, so the sequence replays identically on every rank and on a
    warm start. None outside a warmup, which keeps the key of every other
    compiled module exactly as it was.
    """
    return f"trace{_session.traces}" if _session is not None else None


def retrace(model: torch.nn.Module, vllm_config: VllmConfig) -> None:
    """Give the model's compiled modules a fresh, untraced torch.compile wrapper.

    ``VllmBackend`` refuses a second call ("VllmBackend can only be called
    once"), so another trace needs another backend, and upstream's
    ``reset_compile_wrapper`` builds one. It also restores the first-compile
    path in ``decorators.__call__``, so ``_mark_dynamic_inputs`` runs again and
    the new trace is dynamic like the first.

    Takes the whole model rather than a backbone: ``support_torch_compile``
    puts a wrapper wherever it is applied, which for a per-layer model is every
    decoder layer, and the wrappers are what hold the bytecode hooks that have
    to be dropped before they are replaced.
    """
    from vllm.compilation.wrapper import (
        TorchCompileWithNoGuardsWrapper,
        reset_compile_wrapper,
    )
    from vllm.config import set_current_vllm_config

    if _session is not None:
        _session.traces += 1
        # The refusals belong to the traces being replaced.
        _session.refused.clear()

    wrappers = (
        [model]
        if isinstance(model, TorchCompileWithNoGuardsWrapper)
        else [
            m for m in model.modules() if isinstance(m, TorchCompileWithNoGuardsWrapper)
        ]
    )
    for wrapper in wrappers:
        wrapper.cleanup()  # drop its bytecode hook before replacing it

    # reset_compile_wrapper blanks the cache dirs, which the next backend call
    # recomputes from the files Dynamo inlined -- a trace that inlines a
    # different set lands somewhere else and the hand-over misses. Pin them,
    # as profile_run does around its own reset.
    config = vllm_config.compilation_config
    saved = (config.cache_dir, config.local_cache_dir)
    with set_current_vllm_config(vllm_config):
        reset_compile_wrapper(model)
    config.cache_dir, config.local_cache_dir = saved


# ---------------------------------------------------------------------------
# Tracing every group before compiling any bucket (TPU_PRECOMPILE_TRACE_FIRST)
# ---------------------------------------------------------------------------


class DeferredTraces:
    """The piecewise backends traced while their compiles are deferred.

    In a :func:`warmup`, each trace compiles the buckets its guards accept
    before the ladder reaches the next group and traces again, so the groups
    compile one after another and at most a group's worth of buckets is ever
    in flight. Under :func:`trace_only` a backend records its buckets instead
    (:func:`defer`), the ladder traces every group back to back, and
    :meth:`compile` then compiles every bucket of every trace in one pass.
    Each bucket still compiles exactly once and is handed between the traces
    as before.
    """

    def __init__(self) -> None:
        self.backends: list[Any] = []
        # (subgraph index, bucket) pairs some trace already accepted: a later
        # trace that refuses one does not call for another trace.
        self.accepted: set[tuple[int, int]] = set()

    def compile(self) -> None:
        """Compile every deferred bucket (see ``_patch_vllm_compile_all_ranges``)."""
        if _deferred_compiler is None:
            raise RuntimeError(
                "TPU_PRECOMPILE_TRACE_FIRST needs TPU_PARALLEL_PRECOMPILE=1, which "
                "installs the compiler for deferred traces"
            )
        _deferred_compiler(self.backends)


_deferred: DeferredTraces | None = None
_deferred_compiler: Callable[[list[Any]], None] | None = None


def register_deferred_compiler(fn: Callable[[list[Any]], None]) -> None:
    global _deferred_compiler
    _deferred_compiler = fn


@contextlib.contextmanager
def trace_only() -> Iterator[DeferredTraces]:
    """Scope in which a traced backend defers its compiles (see DeferredTraces)."""
    global _deferred
    assert _session is not None, "trace_only() runs inside warmup()"
    assert _deferred is None, "nested trace_only()"
    _deferred = DeferredTraces()
    try:
        yield _deferred
    finally:
        _deferred = None


def deferring() -> bool:
    """Whether a backend traced now should defer its compiles."""
    return _deferred is not None


def defer(backend: Any) -> None:
    """Record a freshly traced piecewise backend instead of compiling it.

    A bucket the trace's guards accept is left for :meth:`DeferredTraces.compile`
    to build; one they refuse is left to be handed over from the trace that
    builds it, and counts as refused (so the ladder traces again) unless an
    earlier trace already accepted it. Every entry runs a placeholder until
    then, so the warmup forward that triggered the trace completes without
    executing anything on the device.
    """
    assert _deferred is not None and _session is not None
    shape_env = trace_shape_env(backend.graph)
    placeholder = _placeholder_runnable(backend.graph, backend.sym_shape_indices)
    for entry in backend.range_entries.values():
        if entry.compiled:
            continue
        size = (
            entry.compile_range.start if entry.compile_range.is_single_size() else None
        )
        subgraph_bucket = (
            (backend.piecewise_compile_index, size) if size is not None else None
        )
        if subgraph_bucket is not None and subgraph_bucket in _deferred.accepted:
            # An earlier trace builds it; hand it over, as runnable_for does.
            entry._tpu_deferred = "hand_over"
        elif size is None or unsupported_reason(shape_env, size) is None:
            entry._tpu_deferred = "compile"
            if subgraph_bucket is not None:
                _deferred.accepted.add(subgraph_bucket)
        else:
            entry._tpu_deferred = "hand_over"
            _session.refused.add(size)
        entry.runnable = placeholder(size)
        entry.compiled = True
    backend._tpu_trace = _session.traces
    _deferred.backends.append(backend)


def refusing_runnable(key: str) -> Callable[..., Any]:
    """The runnable of a bucket an earlier, replaced trace refused: never run."""

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ShapeSpecializationError(
            f"{key} belongs to a replaced trace and was never compiled there"
        )

    return refuse


def _placeholder_runnable(
    graph: fx.GraphModule, sym_shape_indices: Sequence[int]
) -> Callable[[int | None], Callable[..., Any]]:
    """A stand-in for a deferred bucket: zeros in the traced output's shapes.

    The output shapes are the trace's own, with every symbol taking the token
    count -- the substitution :func:`unsupported_reason` and vLLM's
    ``create_concrete_args`` use. Only the warmup forward that triggered the
    trace runs it; its outputs are discarded.
    """
    from torch.utils import _pytree as pytree

    output = next(node for node in graph.graph.nodes if node.op == "output")
    leaves, spec = pytree.tree_flatten(output.args[0])
    metas = [
        _placeholder_value(leaf) if isinstance(leaf, fx.Node) else leaf
        for leaf in leaves
    ]
    shape_env = trace_shape_env(graph)
    symbols = set(shape_env.var_to_range) if shape_env is not None else set()

    def concrete(dim: Any, substitution: dict) -> int:
        if isinstance(dim, torch.SymInt):
            value = sympy.sympify(dim.node.expr).xreplace(substitution)
            try:
                return int(value)
            except TypeError as e:
                raise ShapeSpecializationError(
                    f"deferred trace output dimension {dim} stays symbolic ({value})"
                ) from e
        return int(dim)

    def for_size(size: int | None) -> Callable[..., Any]:
        def run(*args: Any) -> Any:
            tokens = int(args[sym_shape_indices[0]]) if sym_shape_indices else size
            substitution = {symbol: tokens for symbol in symbols}
            device = next(
                (arg.device for arg in args if isinstance(arg, torch.Tensor)), None
            )
            values: list[Any] = []
            for meta in metas:
                if isinstance(meta, torch.Tensor):
                    shape = [concrete(dim, substitution) for dim in meta.shape]
                    values.append(torch.zeros(shape, dtype=meta.dtype, device=device))
                elif isinstance(meta, torch.SymInt):
                    values.append(concrete(meta, substitution))
                else:
                    values.append(meta)
            return pytree.tree_unflatten(values, spec)

        return run

    return for_size


def _entry_cache_key(backend: Any, entry: Any) -> tuple[str | None, str, int] | None:
    r = entry.compile_range
    if not r.is_single_size():
        return None
    compiler = getattr(
        getattr(backend.vllm_backend, "compiler_manager", None), "compiler", None
    )
    key = (
        f"artifact_compile_range_{r.start}_{r.end}"
        f"_subgraph_{backend.piecewise_compile_index}"
    )
    return getattr(compiler, "cache_dir", None), key, r.start


def check_deferred_handover(
    to_compile: Sequence[tuple[Any, Any]],
    to_hand_over: Sequence[tuple[Any, Any]],
    live_trace: int,
) -> None:
    """Validate deferred bucket hand-overs upfront before running XLA compiles."""
    dry = BucketExecutables()
    for backend, entry in to_compile:
        resolved = _entry_cache_key(backend, entry)
        if resolved is not None:
            cache_dir, key, _ = resolved
            dry.put(cache_dir, key, backend.graph, lambda *a: a)
    for backend, entry in to_hand_over:
        if getattr(backend, "_tpu_trace", 0) != live_trace:
            continue
        resolved = _entry_cache_key(backend, entry)
        if resolved is not None:
            cache_dir, key, start = resolved
            dry.get(cache_dir, key, backend.graph, start)
