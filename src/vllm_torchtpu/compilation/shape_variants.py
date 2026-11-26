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

Standard shape specialization does no serving-time dispatch: vLLM's own
``PiecewiseBackend.__call__`` picks ``range_entries[Range(n, n)].runnable``,
and each bucket is compiled once. When the opt-in decode-only overlay
(``TPU_RPA_DECODE_ONLY_GRAPH=1``) is attached to decode-sized buckets,
:func:`decode_only_runnable` checks at serving dispatch whether the active
wrapper context requests the decode-only executable for that bucket, adding
one TPU program region per covered decode bucket.
Only the Dynamo traces multiply, by one per structurally distinct group.

The partition follows from the guards alone, so every DP/TP rank derives the
same one and stays in lockstep through the collectives inside the specialized
region. The decision has to come from a tensor shape: that is what leaves a
guard behind. A branch on some other python value is guarded outside the
ShapeEnv, vLLM drops that guard, and nothing here can see it.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

import sympy
import torch
import torch.fx as fx

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    is_rpa_decode_only_active,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ShapeSpecializationError(RuntimeError):
    """A token bucket cannot be served by the graph asked to compile it."""


class _UncompiledRunnable:
    """Placeholder for a bucket refused or skipped by a trace; never executed."""

    def __init__(self, message: str) -> None:
        self._message = message

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ShapeSpecializationError(self._message)


# ---------------------------------------------------------------------------
# What a trace assumed about the token count
# ---------------------------------------------------------------------------


def _placeholder_value(node: fx.Node) -> Any:
    return node.meta.get("example_value", node.meta.get("val"))


def _placeholder_values(graph: fx.GraphModule) -> list[Any]:
    values = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            break
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


def _signature(graph: fx.GraphModule) -> tuple:
    """The submodule input list a handed-over runnable has to match.

    Symbolic dims compare as "?", so the token dimension matches across traces
    but a structural difference -- a split graph whose subgraph takes one chunk
    in one variant and two in another -- does not.
    """

    def shape(value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return type(value).__name__
        dims = tuple(
            "?" if isinstance(d, torch.SymInt) else int(d) for d in value.shape
        )
        return (str(value.dtype), dims)

    return tuple(shape(value) for value in _placeholder_values(graph))


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

    def __init__(
        self,
        *,
        allowed_sizes: frozenset[int] | None = None,
        cache_tag: str | None = None,
    ) -> None:
        self._runnables: dict[
            tuple[str | None, str], tuple[Callable[..., Any], tuple]
        ] = {}
        self.refused: set[int] = set()
        self.traces = 0
        self.allowed_sizes = allowed_sizes
        self.cache_tag = cache_tag
        self.backends: list[tuple[int, Any]] = []

    def get(
        self, cache_dir: str | None, key: str, graph: fx.GraphModule
    ) -> Callable[..., Any] | None:
        found = self._runnables.get((cache_dir, key))
        if found is None:
            return None
        runnable, signature = found
        if signature != _signature(graph):
            raise ShapeSpecializationError(
                f"{key} was compiled by another trace of this model whose "
                f"inputs differ ({signature} vs {_signature(graph)}). The "
                "graph is split, so its subgraph boundaries move with the "
                "specialization and a bucket cannot be handed between traces. "
                "The split comes from compilation_config.splitting_ops "
                "(boundaries declared by the model's config hook or the "
                "caller, or the RPA boundaries TPU_KERNEL_ITER_MODE adds); "
                "drop those boundaries to serve this model."
            )
        return runnable

    def put(
        self,
        cache_dir: str | None,
        key: str,
        graph: fx.GraphModule,
        runnable: Callable[..., Any],
    ) -> None:
        self._runnables[(cache_dir, key)] = (runnable, _signature(graph))

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
def warmup(
    *,
    allowed_sizes: frozenset[int] | None = None,
    cache_tag: str | None = None,
) -> Iterator[BucketExecutables]:
    """Scope in which traces of one model may hand buckets to one another."""
    global _session
    assert _session is None, "nested shape-specialization warmups"
    _session = BucketExecutables(allowed_sizes=allowed_sizes, cache_tag=cache_tag)
    try:
        yield _session
    finally:
        _session = None


def skipped_runnable(key: str, size: int | None, tag: str | None) -> Callable[..., Any]:
    """Placeholder for a bucket excluded by a variant's allowed_sizes."""
    return _UncompiledRunnable(f"{key} (bucket {size}) was skipped by variant {tag!r}")


def register_backend(backend: Any) -> None:
    """Record a PiecewiseBackend instantiated during an active warmup session."""
    if _session is not None:
        _session.backends.append((_session.traces, backend))


def collect_variant_runnables(
    session: BucketExecutables,
    *,
    prefix_suffix: str = "",
) -> dict[tuple[str, int, int], tuple[Callable[..., Any], tuple]]:
    """Collect compiled `(base_prefix, subgraph_idx, size)` variant entries."""
    collected: dict[tuple[str, int, int], tuple[Callable[..., Any], tuple]] = {}
    for _, backend in session.backends:
        assert backend.graph is not None, (
            "collect_variant_runnables requires an FX graph (mega-AOT is unsupported)"
        )
        prefix = backend.vllm_backend.prefix
        base_prefix = prefix.removesuffix(prefix_suffix) if prefix_suffix else prefix
        sig = _signature(backend.graph)
        for entry in backend.range_entries.values():
            if not entry.compiled or not entry.compile_range.is_single_size():
                continue
            runnable = entry.runnable
            if runnable is None or isinstance(runnable, _UncompiledRunnable):
                continue
            size = entry.compile_range.start
            if session.allowed_sizes is not None and size not in session.allowed_sizes:
                continue
            collected[(base_prefix, backend.piecewise_compile_index, size)] = (
                runnable,
                sig,
            )
    return collected


def attach_decode_only_runnables(
    session: BucketExecutables,
    decode_runnables: dict[tuple[str, int, int], tuple[Callable[..., Any], tuple]],
) -> int:
    """Attach decode-only runnables onto live PiecewiseBackends with sig check."""
    attached = 0
    for trace_idx, backend in session.backends:
        if trace_idx != session.traces:
            continue
        assert backend.graph is not None, (
            "attach_decode_only_runnables requires an FX graph "
            "(mega-AOT is unsupported)"
        )
        prefix = backend.vllm_backend.prefix
        sig = _signature(backend.graph)
        per_size: dict[int, Callable[..., Any]] = {}
        for entry in backend.range_entries.values():
            if not entry.compile_range.is_single_size():
                continue
            size = entry.compile_range.start
            found = decode_runnables.get(
                (prefix, backend.piecewise_compile_index, size)
            )
            if found is None:
                continue
            runnable, decode_sig = found
            if decode_sig != sig:
                raise ShapeSpecializationError(
                    f"decode-only graph for subgraph {backend.piecewise_compile_index} "
                    f"bucket {size} has mismatched inputs ({decode_sig} vs {sig})"
                )
            per_size[size] = runnable
            attached += 1
        if per_size:
            _decode_only_runnables[backend] = per_size
    if attached != len(decode_runnables):
        raise ShapeSpecializationError(
            f"attached {attached} decode-only entries out of "
            f"{len(decode_runnables)} compiled"
        )
    return attached


_decode_only_runnables: weakref.WeakKeyDictionary[
    Any, dict[int, Callable[..., Any]]
] = weakref.WeakKeyDictionary()
_base_compile_prefixes: weakref.WeakKeyDictionary[torch.nn.Module, str] = (
    weakref.WeakKeyDictionary()
)


def decode_only_runnable(
    backend: Any, runtime_shape: int | None
) -> Callable[..., Any] | None:
    """Return `backend`'s decode-only runnable for `runtime_shape` when active."""
    if runtime_shape is None or not _decode_only_runnables:
        return None
    decode_runnables = _decode_only_runnables.get(backend)
    if decode_runnables is None or not is_rpa_decode_only_active():
        return None
    return decode_runnables.get(runtime_shape)


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
    if (
        _session is not None
        and _session.allowed_sizes is not None
        and size is not None
        and size not in _session.allowed_sizes
    ):
        return skipped_runnable(key, size, _session.cache_tag)

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

    return _UncompiledRunnable(
        f"{key} was left to a later trace ({reason}) and never compiled; "
        "the warmup ladder is out of step with the partition."
    )


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
    if _session is None:
        return None
    if _session.cache_tag:
        return f"{_session.cache_tag}_trace{_session.traces}"
    return f"trace{_session.traces}"


def retrace(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    *,
    prefix_suffix: str = "",
) -> None:
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
        if prefix_suffix or wrapper in _base_compile_prefixes:
            wrapper_any = cast(Any, wrapper)
            base_prefix = _base_compile_prefixes.setdefault(
                wrapper, cast(str, wrapper_any._compile_prefix)
            )
            wrapper_any._compile_prefix = (
                f"{base_prefix}{prefix_suffix}" if prefix_suffix else base_prefix
            )

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
        if (
            _session.allowed_sizes is not None
            and size is not None
            and size not in _session.allowed_sizes
        ):
            entry._tpu_deferred = "skip"
            entry._tpu_skip_tag = _session.cache_tag
            entry.runnable = placeholder(size)
            entry.compiled = True
            continue
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
    return _UncompiledRunnable(
        f"{key} belongs to a replaced trace and was never compiled there"
    )


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
