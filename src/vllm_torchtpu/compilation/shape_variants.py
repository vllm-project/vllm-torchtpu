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
import os
import re
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


# One compiled range can serve many decoder layers, so a layer index in a state
# path is not part of the state's identity.
_MODULE_INDEX = re.compile(r"_modules_\d+_")


def _identities(graph: fx.GraphModule) -> tuple:
    """Per placeholder (aligned with :func:`_signature`): which module state it
    is, else None.

    Dynamo lifts parameters and buffers as placeholders named after their path
    (``l_self_modules_..._modules_attn_hyper_connection_buffers__up_t_``);
    activations get trace-local names and match by shape only. Shapes alone
    cannot tell two same-shaped weights apart, so a trace that takes them in
    the other order would be handed a bucket with the two exchanged (t128:
    HC down and the kept up transpose, both [320, 10240]).
    """
    out = []
    for node in graph.graph.nodes:
        if node.op != "placeholder":
            break
        if node.meta.get("example_value", node.meta.get("val")) is None:
            continue
        name = str(node.target)
        if "_parameters_" in name or "_buffers_" in name:
            out.append(_MODULE_INDEX.sub("_modules_#_", name))
        else:
            out.append(None)
    return tuple(out)


def _identity_mismatch(old: tuple, new: tuple, perm: list[int] | None) -> str | None:
    """The first module state the new trace takes where the old one took
    another, or None (positions mapped through ``perm`` when given)."""
    if len(old) != len(new):
        return f"{len(old)} vs {len(new)} placeholders"
    for o, ident in enumerate(old):
        n = perm[o] if perm is not None else o
        if ident != new[n]:
            return f"input {o}: {ident} vs {new[n]}"
    return None


def _check_identities(key: str, old: tuple, new: tuple, perm: list[int] | None) -> None:
    mismatch = _identity_mismatch(old, new, perm)
    if mismatch is None:
        return
    message = (
        f"{key} was compiled by another trace of this model that takes its "
        f"module state in another order ({mismatch}); handing it over would "
        "bind those inputs exchanged."
    )
    if os.environ.get("VLLM_TPU_SHAPE_VARIANT_IDENTITY_WARN") == "1":
        logger.warning("[shape-variants] %s (VLLM_TPU_SHAPE_VARIANT_IDENTITY_WARN=1: "
                       "handing over anyway)", message)
        return
    raise ShapeSpecializationError(message)


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
    """:func:`_signature` and :func:`_identities`, for callers outside this
    module.

    Stable across processes -- dtypes as strings, dims as ints or "?", state
    paths as strings -- so a compile cache written by one run can check that
    the graph it is being replayed against still takes the same inputs, the
    same module state in the same order included.
    """
    return _signature(graph) + (("identities",) + _identities(graph),)


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
            tuple[str | None, str], tuple[Callable[..., Any], tuple, tuple, tuple]
        ] = {}
        self.refused: set[int] = set()
        self.traces = 0

    def get(
        self, cache_dir: str | None, key: str, graph: fx.GraphModule
    ) -> Callable[..., Any] | None:
        found = self._runnables.get((cache_dir, key))
        if found is None:
            return None
        runnable, signature, sources, identities = found
        new_signature = _signature(graph)
        if signature == new_signature:
            _check_identities(key, identities, _identities(graph), None)
            return runnable
        new_sources = _symint_sources(graph)
        perm = symint_permutation((signature, sources), (new_signature, new_sources))
        if perm is not None:
            _check_identities(key, identities, _identities(graph), perm)
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
            _identities(graph),
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
