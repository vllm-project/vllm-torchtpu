# SPDX-License-Identifier: Apache-2.0
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
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import sympy
import torch
import torch.fx as fx

from vllm_torchtpu.compilation import bucket_filter
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ShapeSpecializationError(RuntimeError):
    """A token bucket cannot be served by the graph asked to compile it."""


# ---------------------------------------------------------------------------
# What a trace assumed about the token count
# ---------------------------------------------------------------------------


def _placeholder_values(graph: fx.GraphModule) -> list[Any]:
    values = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            break
        value = node.meta.get("example_value", node.meta.get("val"))
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


def _symint_sources(graph: fx.GraphModule) -> tuple:
    """Per placeholder: where a SymInt's value comes from, else None.

    Aligned with :func:`_signature`. Symbol names are allocation order within
    one trace's ShapeEnv (the token count is ``s16`` in one trace and ``s72``
    in the next), so a SymInt is identified by the Dynamo source of its root
    symbol instead -- ``L['input_ids'].size()[0]`` in every trace of the same
    forward -- together with its expression in that root. The expression is
    resolved through the ShapeEnv's replacements first, so a derived or
    guard-introduced symbol maps to the root it stands for. A SymInt that does
    not resolve to exactly one sourced root symbol maps to None, which only
    ever matches by position.
    """
    env = trace_shape_env(graph)
    root_name = sympy.Symbol("root", integer=True)

    def source(value: Any) -> Any:
        if not isinstance(value, torch.SymInt) or env is None:
            return None
        expr = env.replace(value.node.expr)
        free = expr.free_symbols
        if len(free) != 1:
            return None
        (root,) = free
        sources = env.var_to_sources.get(root)
        if not sources:
            return None
        name = sources[0].name  # a property in recent torch, a method before
        return (name() if callable(name) else name, str(expr.subs(root, root_name)))

    return tuple(source(value) for value in _placeholder_values(graph))


def _tensor_source(node: fx.Node) -> str | None:
    """A trace-independent name for a tensor placeholder, or None if it has none.

    Dynamo names a graph input after its source (``L['self']...weight`` becomes
    ``l_self_modules_..._parameters_weight_``), and a split subgraph names each
    placeholder after the outer node it reads, so a sourced input -- a
    parameter, a buffer, a forward argument -- carries the same name in every
    trace of the same forward. Intermediate values are named by op order
    (``add_3``), which differs between traces, so they get None and are never
    compared.
    """
    grapharg = node.meta.get("grapharg")
    source = getattr(grapharg, "source", None)
    if source is not None:
        name = source.name  # a property in recent torch, a method before
        return str(name() if callable(name) else name)
    name = str(node.name)
    return name if name.startswith(("l_", "L_")) else None


def _tensor_sources(graph: fx.GraphModule) -> tuple:
    """Per placeholder: :func:`_tensor_source` for a tensor, else None.

    Aligned with :func:`_signature` (placeholders without a value are skipped
    the same way).
    """
    sources = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            break
        value = node.meta.get("example_value", node.meta.get("val"))
        if value is None:
            continue
        sources.append(_tensor_source(node) if isinstance(value, torch.Tensor) else None)
    return tuple(sources)


def source_order_diffs(
    old: tuple[tuple, tuple], new: tuple[tuple, tuple]
) -> list[tuple[int, str | None, str | None]]:
    """Tensor inputs a hand-over would bind to the wrong placeholder.

    ``old`` and ``new`` are (signature, tensor sources) of the trace that
    compiled the runnable and of the trace adopting it. The runnable takes the
    non-SymInt inputs by position (SymInts are permuted by source, see
    :func:`symint_permutation`), so the non-SymInt inputs are compared in
    order. A position counts when both sides name a source and the names
    differ, or when one source sits at different positions in the two traces.
    Returns (non-SymInt position, compiled-for source, given source) entries;
    empty when nothing named is out of place.
    """
    (old_sig, old_srcs), (new_sig, new_srcs) = old, new
    old_t = [old_srcs[i] for i, v in enumerate(old_sig) if v != "SymInt"]
    new_t = [new_srcs[i] for i, v in enumerate(new_sig) if v != "SymInt"]
    diffs = {}
    for j, (o, n) in enumerate(zip(old_t, new_t)):
        if o is not None and n is not None and o != n:
            diffs[j] = (j, o, n)
    new_pos = {n: j for j, n in enumerate(new_t) if n is not None}
    for j, o in enumerate(old_t):
        if o is not None and o in new_pos and new_pos[o] != j and j not in diffs:
            diffs[j] = (j, o, new_t[j] if j < len(new_t) else None)
    return [diffs[j] for j in sorted(diffs)]


def ambiguous_inputs(signature: tuple, sources: tuple) -> tuple[list, list]:
    """Unsourced tensor inputs whose binding the source-order check cannot verify.

    A handed-over runnable takes its inputs by position on a (dtype, shape)
    signature, so a positional mix-up can only exchange inputs of one
    (dtype, shape). Within such a group the sourced inputs are checked by
    name, and a single unsourced input is then forced into the remaining
    place. Two or more unsourced inputs in one group (split-boundary
    activations, named by op order) could trade places unseen.

    Returns (ambiguous, shared): ``ambiguous`` lists (index, signature entry)
    for every unsourced input in a group holding two or more unsourced inputs;
    ``shared`` lists every unsourced input that shares its (dtype, shape) with
    any other input (the looser count).
    """
    groups: dict = {}
    for i, (v, s) in enumerate(zip(signature, sources)):
        if isinstance(v, tuple):  # tensors only: (dtype, dims)
            groups.setdefault(v, []).append((i, s))
    ambiguous, shared = [], []
    for v, members in groups.items():
        unsourced = [i for i, s in members if s is None]
        if len(members) >= 2:
            shared.extend((i, v) for i in unsourced)
        if len(unsourced) >= 2:
            ambiguous.extend((i, v) for i in unsourced)
    return sorted(ambiguous), sorted(shared)


def symint_permutation(
    old: tuple[tuple, tuple], new: tuple[tuple, tuple]
) -> list[int] | None:
    """For each input of the old trace, where it sits among the new one's.

    Two traces of one split graph can take the same inputs and differ only in
    where a SymInt placeholder sits: FX orders a subgraph's placeholders by first
    use, and a specialized branch can first touch the token count later. Returns
    None unless that is the only difference: every non-SymInt input in the same
    order, and the SymInts matched one to one by source (see
    :func:`_symint_sources`).
    """
    (old_sig, old_srcs), (new_sig, new_srcs) = old, new
    if len(old_sig) != len(new_sig):
        return None
    old_rest = [i for i, v in enumerate(old_sig) if v != "SymInt"]
    new_rest = [i for i, v in enumerate(new_sig) if v != "SymInt"]
    if [old_sig[i] for i in old_rest] != [new_sig[i] for i in new_rest]:
        return None
    old_ints = [i for i, v in enumerate(old_sig) if v == "SymInt"]
    new_ints = [i for i, v in enumerate(new_sig) if v == "SymInt"]
    keys = [old_srcs[i] for i in old_ints]
    new_by_key = {new_srcs[i]: i for i in new_ints}
    if (
        any(key is None for key in keys)
        or None in new_by_key
        or len(set(keys)) != len(keys)
        or len(new_by_key) != len(new_ints)
        or set(keys) != set(new_by_key)
    ):
        return None
    perm = [0] * len(old_sig)
    for o, n in zip(old_rest, new_rest):
        perm[o] = n
    for o in old_ints:
        perm[o] = new_by_key[old_srcs[o]]
    return perm


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
            tuple[str | None, str], tuple[Callable[..., Any], tuple, tuple]
        ] = {}
        self.refused: set[int] = set()
        self.traces = 0
        # Source-order check (log only): tensor sources per stored runnable,
        # and counts reported when the warmup ends.
        self._tensor_srcs: dict[tuple[str | None, str], tuple | None] = {}
        self.source_handovers = 0
        self.source_compared = 0
        self.source_blind = 0
        self.source_mismatches = 0
        self.source_ambiguous = 0
        self.source_ambiguous_inputs = 0
        self.source_shared_inputs = 0

    def _check_source_order(
        self,
        cache_dir: str | None,
        key: str,
        signature: tuple,
        new_signature: tuple,
        graph: fx.GraphModule,
    ) -> None:
        """Log a hand-over whose tensor inputs would bind out of place.

        Diagnostics only: never raises, never changes what :meth:`get`
        returns, runs at compile time on the FX graph (nothing is traced).
        """
        self.source_handovers += 1
        try:
            old_srcs = self._tensor_srcs.get((cache_dir, key))
            new_srcs = _tensor_sources(graph)
            ambiguous, shared = ambiguous_inputs(new_signature, new_srcs)
            self.source_shared_inputs += len(shared)
            if ambiguous:
                self.source_ambiguous += 1
                self.source_ambiguous_inputs += len(ambiguous)
                logger.info(
                    "[shape-variants] SOURCE-ORDER AMBIGUOUS %s handed to trace %d: "
                    "%d unsourced input(s) share a (dtype, shape) with another "
                    "unsourced input, so their binding is not verified "
                    "(input index, (dtype, shape)): %s",
                    key,
                    self.traces,
                    len(ambiguous),
                    ambiguous,
                )
            if old_srcs is None or not any(old_srcs) or not any(new_srcs):
                self.source_blind += 1
                return
            self.source_compared += 1
            diffs = source_order_diffs((signature, old_srcs), (new_signature, new_srcs))
            if diffs:
                self.source_mismatches += 1
                logger.warning(
                    "[shape-variants] SOURCE-ORDER MISMATCH %s handed to trace %d: "
                    "tensor inputs are bound by position and these differ "
                    "(input index, compiled-for source, given source): %s | "
                    "compiled-for sources: %s | given sources: %s",
                    key,
                    self.traces,
                    diffs,
                    [(i, s) for i, s in enumerate(old_srcs) if s is not None],
                    [(i, s) for i, s in enumerate(new_srcs) if s is not None],
                )
        except Exception as exc:  # pragma: no cover - diagnostics must not fail a compile
            self.source_blind += 1
            logger.warning(
                "[shape-variants] source-order check skipped for %s: %r", key, exc
            )

    def get(
        self, cache_dir: str | None, key: str, graph: fx.GraphModule
    ) -> Callable[..., Any] | None:
        found = self._runnables.get((cache_dir, key))
        if found is None:
            return None
        runnable, signature, sources = found
        new_signature = _signature(graph)
        if signature == new_signature:
            self._check_source_order(cache_dir, key, signature, new_signature, graph)
            return runnable
        new_sources = _symint_sources(graph)
        perm = symint_permutation((signature, sources), (new_signature, new_sources))
        if perm is not None:
            self._check_source_order(cache_dir, key, signature, new_signature, graph)
            moved = [
                (o, n, sources[o]) if sources[o] is not None else (o, n)
                for o, n in enumerate(perm)
                if o != n
            ]
            logger.info(
                "[shape-variants] %s adopted by trace %d with SymInt placeholders "
                "permuted (old index <- new index[, source]): %s",
                key,
                self.traces,
                moved,
            )

            def adopted(*args: Any, **kwargs: Any) -> Any:
                return runnable(*[args[n] for n in perm], **kwargs)

            return adopted
        # Name what differs before refusing, so a refusal localizes itself:
        # each trace's SymInt sources (None = no single sourced root, which is
        # what stops symint_permutation) and tensor sources, by input index.
        try:
            old_t = self._tensor_srcs.get((cache_dir, key)) or ()
            new_t = _tensor_sources(graph)
            logger.warning(
                "[shape-variants] REFUSED %s for trace %d: SymInt sources "
                "compiled-for %s given %s | tensor sources compiled-for %s given %s",
                key,
                self.traces,
                [(i, s) for i, (v, s) in enumerate(zip(signature, sources)) if v == "SymInt"],
                [(i, s) for i, (v, s) in enumerate(zip(new_signature, new_sources)) if v == "SymInt"],
                [(i, s) for i, s in enumerate(old_t) if s is not None],
                [(i, s) for i, s in enumerate(new_t) if s is not None],
            )
        except Exception:  # pragma: no cover - diagnostics must not mask the refusal
            pass
        raise ShapeSpecializationError(
            f"{key} was compiled by another trace of this model whose "
            f"inputs differ ({signature} vs {new_signature}). The "
            "graph is split, so its subgraph boundaries move with the "
            "specialization and a bucket cannot be handed between traces; "
            "unset TPU_KERNEL_ITER_MODE to serve this model."
        )

    def put(
        self,
        cache_dir: str | None,
        key: str,
        graph: fx.GraphModule,
        runnable: Callable[..., Any],
    ) -> None:
        self._runnables[(cache_dir, key)] = (
            runnable,
            _signature(graph),
            _symint_sources(graph),
        )
        try:
            self._tensor_srcs[(cache_dir, key)] = _tensor_sources(graph)
        except Exception:  # pragma: no cover - diagnostics must not fail a compile
            self._tensor_srcs[(cache_dir, key)] = None

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
        done, _session = _session, None
        logger.info(
            "[shape-variants] SOURCE-ORDER CHECK handovers=%d compared=%d "
            "blind=%d mismatches=%d traces=%d ambiguous=%d ambiguous_inputs=%d "
            "shared_inputs=%d",
            done.source_handovers,
            done.source_compared,
            done.source_blind,
            done.source_mismatches,
            done.traces,
            done.source_ambiguous,
            done.source_ambiguous_inputs,
            done.source_shared_inputs,
        )


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


def retrace(
    model: torch.nn.Module, vllm_config: VllmConfig, size: int | None = None
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

    With ``size``, only the wrappers serving that bucket are replaced (see
    compilation/bucket_filter): a model routing buckets to two compiled modules
    retraces one without dropping the other's graph, which the ladder may
    already be past.
    """
    from vllm.compilation.wrapper import (
        TorchCompileWithNoGuardsWrapper,
        reset_compile_wrapper,
    )
    from vllm.config import set_current_vllm_config

    wrappers = (
        [model]
        if isinstance(model, TorchCompileWithNoGuardsWrapper)
        else [
            m for m in model.modules() if isinstance(m, TorchCompileWithNoGuardsWrapper)
        ]
    )
    if size is not None:
        wrappers = [w for w in wrappers if bucket_filter.module_serves(w, size)]

    if _session is not None:
        _session.traces += 1
        # The refusals belong to the traces being replaced.
        _session.refused = {
            refused
            for refused in _session.refused
            if not any(bucket_filter.module_serves(w, refused) for w in wrappers)
        }

    for wrapper in wrappers:
        wrapper.cleanup()  # drop its bytecode hook before replacing it

    # reset_compile_wrapper blanks the cache dirs, which the next backend call
    # recomputes from the files Dynamo inlined -- a trace that inlines a
    # different set lands somewhere else and the hand-over misses. Pin them,
    # as profile_run does around its own reset.
    config = vllm_config.compilation_config
    saved = (config.cache_dir, config.local_cache_dir)
    with set_current_vllm_config(vllm_config):
        for wrapper in wrappers:
            reset_compile_wrapper(wrapper)
    config.cache_dir, config.local_cache_dir = saved
