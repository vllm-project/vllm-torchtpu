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

Nothing here runs at serving time beyond a positional reorder of a handed-over
bucket's arguments: dispatch stays vLLM's own
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
import dataclasses
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


# ---------------------------------------------------------------------------
# What a trace assumed about the token count
# ---------------------------------------------------------------------------


def _placeholder_nodes(graph: fx.GraphModule) -> list[fx.Node]:
    nodes = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            break
        nodes.append(node)
    return nodes


def _placeholder_value(node: fx.Node) -> Any:
    return node.meta.get("example_value", node.meta.get("val"))


def _placeholder_values(graph: fx.GraphModule) -> list[Any]:
    values = (_placeholder_value(node) for node in _placeholder_nodes(graph))
    return [value for value in values if value is not None]


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
    """A placeholder's dtype and shape, comparable across traces.

    Symbolic dims compare as "?", so the token dimension matches across traces
    but a structural difference -- a split graph whose subgraph takes one chunk
    in one variant and two in another -- does not.
    """
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return type(value).__name__
    dims = tuple("?" if isinstance(d, torch.SymInt) else int(d) for d in value.shape)
    return (str(value.dtype), dims)


@dataclasses.dataclass(frozen=True)
class _Input:
    """A placeholder as a handover compares and names it, without its graph.

    A stored bucket's inputs live until the warmup ends. The fx.Node itself
    would keep its whole GraphModule -- the compiler's deep copy, with every
    intermediate's metadata -- alive through ``node.graph`` for that long, one
    per subgraph and bucket. The example value and Dynamo's source do not.
    """

    name: str
    value: Any
    source: Any  # torch._dynamo.source.Source, when the graph still has it
    graph_input: bool  # one of Dynamo's graph inputs, named after its source


def _graph_input_source(node: fx.Node) -> Any:
    """The Dynamo source recorded on a graph input, or None. Dynamo deletes
    each placeholder's GraphArg before it calls the backend, so the graphs
    vLLM compiles carry none; only graphs built without that step do."""
    grapharg = node.meta.get("grapharg")
    return grapharg.source if grapharg is not None else None


def _is_graph_input(node: fx.Node) -> bool:
    """Whether a placeholder is one of Dynamo's graph inputs, which Dynamo
    names after their source (``l_self_..._weight_``, the size symbol
    ``s77``), rather than a split subgraph's intermediate, which is named
    after the parent graph's node that computed it.

    torch's ``split_module`` copies a parent placeholder's meta into every
    subgraph that takes it, and an intermediate gets the meta of the node
    that computed it. Dynamo records a tensor input's attributes under
    ``meta["tensor_dict"]`` and nothing computed in the graph has them; an
    int input is the bare size symbol it is named after.
    """
    if node.meta.get("grapharg") is not None:
        return True
    value = _placeholder_value(node)
    if isinstance(value, torch.Tensor):
        return "tensor_dict" in node.meta
    if isinstance(value, torch.SymInt):
        expr = value.node.expr
        return bool(getattr(expr, "is_Symbol", False)) and str(expr) == node.name
    return False


def _inputs(graph: fx.GraphModule) -> list[_Input]:
    return [
        _Input(
            node.name,
            _placeholder_value(node),
            _graph_input_source(node),
            _is_graph_input(node),
        )
        for node in _placeholder_nodes(graph)
    ]


def _signature(graph: fx.GraphModule | Sequence[_Input]) -> tuple:
    """The submodule input list, by dtype and shape."""
    inputs = _inputs(graph) if isinstance(graph, fx.GraphModule) else graph
    return tuple(_meta_entry(i.value) for i in inputs if i.value is not None)


def _handover_inputs(inputs: Sequence[_Input]) -> tuple[tuple[str, Any], ...]:
    """The submodule's arguments in call order, by name, dtype and shape.

    A handed-over runnable takes its arguments positionally, and two traces of
    one model may lift the same buffers in a different order, so the named
    graph inputs among them are matched by name (see :func:`_input_permutation`).
    """
    return tuple((i.name, _meta_entry(i.value)) for i in inputs)


def _input_permutation(
    stored: Sequence[_Input], current: Sequence[_Input]
) -> list[int] | None:
    """Where each stored argument sits in the current call, or None when the
    two traces' inputs differ beyond that.

    Only named graph inputs -- placeholders Dynamo read from a source and named
    after it (``l_self_..._weight_``) -- are matched by name: two traces of one
    model may lift the same buffers in a different order. A split subgraph's
    intermediates are named after the parent graph's node in creation order
    (``getitem_7``), so in another trace the same name can stand for a
    different value; they keep their positions. When the named inputs differ,
    or the intermediates are not at the same positions, every input is matched
    by position, dtype and shape, as on main.
    """
    if len(stored) != len(current):
        return None
    identity = list(range(len(stored)))
    named = [i for i, graph_input in enumerate(stored) if graph_input.graph_input]
    if named == [i for i, graph_input in enumerate(current) if graph_input.graph_input]:
        stored_keys = _handover_inputs(stored)
        current_keys = _handover_inputs(current)
        position = {current_keys[i]: i for i in named}
        perm = list(identity)
        for i in named:
            perm[i] = position.get(stored_keys[i], -1)
        named_set = set(named)
        unnamed_match = all(
            stored_keys[i][1] == current_keys[i][1]
            for i in identity
            if i not in named_set
        )
        if unnamed_match and sorted(perm) == identity:
            return perm
    return identity if _signature(stored) == _signature(current) else None


def _is_int_input(value: Any) -> bool:
    return isinstance(value, (int, torch.SymInt))


def _input_source(graph_input: _Input) -> str | None:
    """Where Dynamo read a graph input from (``L['positions'].size()[0]``),
    when the graph records it, else None: always for a split subgraph's
    intermediate, and for every input of the graphs vLLM compiles, whose
    GraphArgs Dynamo has deleted (an int input's symbol still names its
    sources, see _symbol_source_names)."""
    source = graph_input.source
    return None if source is None else source.name


def _symbol_source_names(value: Any) -> list[str]:
    """Where a SymInt's symbol was read from (``L['positions'].size()[0]``),
    one name per source, from its own trace's ShapeEnv, which keeps them
    after Dynamo has deleted the graph's GraphArgs. Empty for anything else."""
    if not isinstance(value, torch.SymInt):
        return []
    try:
        expr = value.node.expr
        sources = value.node.shape_env.var_to_sources.get(expr)
    except AttributeError:
        return []
    return [s.name for s in sources or ()]


def _symbol_sources(value: Any) -> str | None:
    """The guard sources of a SymInt's symbol, from its trace's ShapeEnv."""
    return ", ".join(_symbol_source_names(value)) or None


def _int_input_sources(graph_input: _Input) -> list[str]:
    """Where an int input was read from: its GraphArg's source when the graph
    still records one, else its symbol's sources."""
    source = _input_source(graph_input)
    if source is not None:
        return [source]
    return _symbol_source_names(graph_input.value)


def _describe_input(graph_input: _Input) -> str:
    value = graph_input.value
    parts = [f"{graph_input.name} {_meta_entry(value)}"]
    if isinstance(value, torch.SymInt):
        parts.append(f"expr={value.node.expr}")
    source = _input_source(graph_input) or _symbol_sources(value)
    if source:
        parts.append(f"from {source}")
    return " ".join(parts)


def _symbol_fate(
    inputs: Sequence[_Input], sources: Sequence[str], trace: str
) -> str | None:
    """What the ShapeEnv of the trace ``inputs`` came from concluded about the
    symbol read from any of ``sources``: its replacement when it was
    specialized, else its value range. ``trace`` names that trace in the
    text."""
    if not sources:
        return None
    shape_env = None
    for graph_input in inputs:
        value = graph_input.value
        if isinstance(value, torch.SymInt):
            shape_env = value.node.shape_env
        else:
            shape_env = getattr(getattr(value, "fake_mode", None), "shape_env", None)
        if shape_env is not None:
            break
    if shape_env is None:
        return None
    for symbol, symbol_sources in shape_env.var_to_sources.items():
        if any(s.name in sources for s in symbol_sources):
            replacement = shape_env.replacements.get(symbol)
            if replacement is not None:
                return f"{symbol} = {replacement} in {trace}"
            value_range = shape_env.var_to_range.get(symbol)
            return f"{symbol} in {value_range} in {trace}"
    return f"no symbol of {trace} reads {', '.join(sources)}"


def _signature_delta(stored: Sequence[Any], current: Sequence[Any]) -> str:
    """The inputs only one of two traces takes, most frequent first."""

    def render(counts: Counter) -> str:
        if not counts:
            return "nothing"
        by_count = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        return ", ".join(f"{count}x {item}" for item, count in by_count)

    return (
        f"{len(stored)} inputs vs {len(current)}. "
        f"Only in the earlier trace: {render(Counter(stored) - Counter(current))}. "
        f"Only in this one: {render(Counter(current) - Counter(stored))}"
    )


_NAMED_INPUTS_SHOWN = 8


def _input_delta(stored: Sequence[_Input], current: Sequence[_Input]) -> str:
    """:func:`_signature_delta` of two traces' inputs, naming the inputs that
    account for it: placeholder, dtype and shape, and the source Dynamo read it
    from, so the model code that lifts one and not the other can be found."""
    stored_entries = _signature(stored)
    current_entries = _signature(current)
    delta = _signature_delta(stored_entries, current_entries)

    def unmatched(side: Sequence[_Input], other: Sequence[_Input]) -> list[_Input]:
        extra = Counter(_signature(side)) - Counter(_signature(other))
        other_keys = set(_handover_inputs(other))
        inputs = []
        for graph_input, key in zip(side, _handover_inputs(side)):
            if key not in other_keys and extra[key[1]] > 0:
                extra[key[1]] -= 1
                inputs.append(graph_input)
        return inputs

    def render(inputs: list[_Input], other: Sequence[_Input], other_trace: str) -> str:
        shown = []
        for graph_input in inputs[:_NAMED_INPUTS_SHOWN]:
            text = _describe_input(graph_input)
            if _is_int_input(graph_input.value):
                fate = _symbol_fate(other, _int_input_sources(graph_input), other_trace)
                if fate:
                    text += f" ({fate})"
            shown.append(text)
        if len(inputs) > _NAMED_INPUTS_SHOWN:
            shown.append(f"... {len(inputs) - _NAMED_INPUTS_SHOWN} more")
        return "; ".join(shown)

    only_stored = unmatched(stored, current)
    only_current = unmatched(current, stored)
    if only_stored:
        named = render(only_stored, current, "this trace")
        delta += f". Named, earlier trace: {named}"
    if only_current:
        named = render(only_current, stored, "the earlier trace")
        delta += f". Named, this trace: {named}"
    return delta


def _handover_error(key: str, delta: str) -> ShapeSpecializationError:
    return ShapeSpecializationError(
        f"{key} was compiled by another trace of this model whose inputs "
        f"cannot be matched to this trace's ({delta}): named graph inputs are "
        "matched by name, the rest by position, dtype and shape. Either the "
        "graph is split, so its subgraph boundaries move with the "
        "specialization, or the two "
        "sides of a token-count branch lift different buffers; a bucket cannot "
        "be handed between such traces. A split comes from "
        "compilation_config.splitting_ops (boundaries declared by the model's "
        "config hook or the caller, or the RPA boundaries TPU_KERNEL_ITER_MODE "
        "adds): drop those boundaries, or give both sides of the branch the "
        "same inputs, to serve this model."
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
            tuple[str | None, str], tuple[Callable[..., Any], list[_Input]]
        ] = {}
        self.refused: set[int] = set()
        self.traces = 0

    def get(
        self, cache_dir: str | None, key: str, graph: fx.GraphModule
    ) -> Callable[..., Any] | None:
        found = self._runnables.get((cache_dir, key))
        if found is None:
            return None
        runnable, stored_inputs = found
        current_inputs = _inputs(graph)
        perm = _input_permutation(stored_inputs, current_inputs)
        if perm is None:
            raise _handover_error(key, _input_delta(stored_inputs, current_inputs))
        if perm == list(range(len(perm))):
            return runnable
        moved = [stored_inputs[i].name for i, j in enumerate(perm) if j != i]
        logger.info(
            "[shape-variants] %s handed over with %d of %d inputs reordered: %s%s",
            key,
            len(moved),
            len(perm),
            ", ".join(moved[:_NAMED_INPUTS_SHOWN]),
            " ..." if len(moved) > _NAMED_INPUTS_SHOWN else "",
        )
        # len(perm) >= 2 here, so itemgetter returns a tuple.
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
        # The inputs' names, values and sources are kept, not just their dtypes
        # and shapes, so a refused handover can name the inputs that differ and
        # where Dynamo read them from.
        self._runnables[(cache_dir, key)] = (runnable, _inputs(graph))

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


def runnable_for(
    cache_dir: str | None, key: str, graph: fx.GraphModule, size: int | None = None
) -> Callable[..., Any] | None:
    """What this compile should use instead of compiling, if anything.

    The load path is screened too: the cache key is (subgraph, bucket) with no
    graph identity, so a graph would otherwise adopt an executable another
    trace compiled and wrap it in its own aot_autograd wrapper. On TPU that
    executes as "Donation requested for invalid buffer".
    Refusing here leaves the bucket to the trace that owns it, which is the
    only one that can rebuild a wrapper matching the executable.
    """
    ready = _session.get(cache_dir, key, graph) if _session else None
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
