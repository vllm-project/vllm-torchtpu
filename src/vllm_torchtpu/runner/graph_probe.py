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
"""Attribute new XLA cache entries to a region of a step.

One counter, shared process-wide. Each probe reports the entries created since
whichever probe ran last, so the per-region numbers partition the total however
the probe points are spread across modules -- a second, module-local counter
would report a delta overlapping its neighbours' and the regions would sum to
more than were actually created.

Lazy execution makes placement matter: an op traced before a probe point is
only counted there if something materialized it first. Probe points that need
an exact boundary call ``vllm_torchtpu.utils.synchronize_tensors`` before
probing (see the mm-encoder boundaries in ``tpu_runner.execute_model``).
"""

import dataclasses
import os
import sys

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

_last = 0
_last_miss = 0
_keys_wanted = False


def total_entries() -> int:
    return len(torch.tpu._get_cache_stats().per_entry_stats)


def total_misses() -> int:
    return int(torch.tpu._get_cache_misses())


def _assert_miss_keys() -> bool:
    """Set dump-on-miss and return what the runtime says it is afterwards."""
    torch.tpu._set_dump_on_cache_miss(True)
    return bool(torch.tpu._get_dump_on_cache_miss())


def _capture_c_stderr(directory: str) -> str:
    """Point this process's fd 2 at its own file, leaving Python's alone.

    The miss keys are written straight to fd 2 by the runtime -- no logger, no
    rank prefix -- so under a multi-process server all eight workers' keys
    interleave into one stream if they arrive at all. Duplicating fd 2 first
    and rebinding ``sys.stderr`` to the duplicate keeps Python's warnings and
    tracebacks on the original pipe, so the server log is unchanged and only
    the C++ writes move.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"miss_keys.{os.getpid()}.log")
    sys.stderr.flush()
    sys.stderr = os.fdopen(os.dup(2), "w", buffering=1)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 2)
    os.close(fd)
    return path


def enable_miss_keys() -> None:
    """Have the compilation cache print a key for every miss.

    ``per_entry_stats`` carries durations and read counts but no name or hash,
    so a probe can say *how many* programs a region made and not *which*. This
    is the one identity the runtime does expose -- one stderr line per miss:

        Dumping StableHLO module due to cache miss for key: CompilationCacheKey{
          shapeless_key=..., dimensions_key=..., compile_options_key=...}

    and the split is exactly the distinction the probes cannot draw. Two misses
    sharing a ``shapeless_key`` but differing in ``dimensions_key`` differ only
    in a shape; a differing ``shapeless_key`` means the traced op graph itself
    changed; an identical key appearing twice means the entry was evicted and
    recompiled. Set ``XLA_FLAGS=--xla_dump_to=DIR`` as well and the StableHLO
    module lands under DIR -- worth doing only for a small workload, since the
    backbone dumps a module per token bucket per rank.

    The read-back goes through the Python logger rather than being assumed,
    because the keys themselves come from C++ on fd 2 and a run that produces
    none is otherwise indistinguishable from a run where the flag never took --
    which is exactly what happened once already. ``probe`` re-asserts the flag
    if it is ever seen off, since whichever mechanism could clear it (a fork, a
    fresh runtime client) would do so after this call.
    """
    global _keys_wanted
    if not envs.TPU_GRAPH_PROBE:
        return
    _keys_wanted = True
    try:
        where = "fd 2"
        if envs.TPU_GRAPH_PROBE_KEY_DIR:
            where = _capture_c_stderr(envs.TPU_GRAPH_PROBE_KEY_DIR)
        # The rank prefix the logger adds is what maps this file back to a
        # rank, so log the path rather than deriving the name from a rank the
        # probe would have to go looking for.
        logger.info("[GRAPH_PROBE] dump-on-cache-miss=%s keys->%s",
                    _assert_miss_keys(), where)
    except Exception as e:  # noqa: BLE001 - diagnostics must not break a run
        logger.warning("[GRAPH_PROBE] could not enable miss keys: %s", e)


def _members(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [(f.name, getattr(obj, f.name, None))
                for f in dataclasses.fields(obj)]
    if isinstance(obj, dict):
        return list(obj.items())
    if hasattr(obj, "__dict__"):
        return sorted(vars(obj).items())
    return []


def _render(value, depth: int) -> str | None:
    """How one member reads in a signature, or None if it cannot key a graph."""
    if isinstance(value, torch.Tensor):
        # dtype as well as shape: both are baked into the compiled program.
        return f"{tuple(value.shape)}:{str(value.dtype).removeprefix('torch.')}"
    if isinstance(value, bool):
        return f"={int(value)}"
    if isinstance(value, int):
        return f"={value}"
    if isinstance(value, str):
        # Strings are here because a jax meta_field is often one -- see
        # AttentionMetadata.sequence_layout_kind, which is static to the jit
        # and so keys the program just as firmly as a shape does.
        return f"={value[:32]}"
    if value is None or isinstance(value, float):
        return None
    if depth <= 0:
        return None
    nested = shape_key(value, depth - 1)
    return f"[{nested}]" if nested else None


def shape_key(obj, depth: int = 2) -> str:
    """A signature of everything about ``obj`` that could key an XLA program.

    Tensor shapes and host-side ints both bake into a program, so a probe that
    reports only the token count cannot tell two programs at the same token
    count apart -- which is exactly the case that needs explaining when one
    bucket compiles four times. Walks a couple of levels of dataclass fields,
    dict values and attributes, printing ``name(shape)`` for tensors and
    ``name=value`` for the scalars.

    Members that render identically are collapsed to one entry with a count,
    which is what keeps a per-layer attention-metadata dict down to a line
    instead of sixty repetitions of the same shapes.
    """
    rendered: dict[str, list[str]] = {}
    for name, value in _members(obj):
        body = _render(value, depth)
        if body is not None:
            rendered.setdefault(body, []).append(str(name))
    parts = []
    for body, names in rendered.items():
        head = names[0] if len(names) == 1 else f"{names[0]}x{len(names)}"
        parts.append(f"{head}{body}")
    return ",".join(parts)


def _field(value) -> object:
    """A probe field's value, resolving callables. Never raises: a broken
    signature should degrade the log line, not the run it is measuring."""
    if not callable(value):
        return value
    try:
        return value()
    except Exception as e:  # noqa: BLE001 - diagnostics must not break a run
        return f"<{type(e).__name__}: {e}>"


def probe(tag: str, **fields) -> None:
    """Log how many cache entries appeared since the previous probe.

    A field may be a zero-argument callable, which is invoked only when the
    probe actually has something to report -- ``shape_key`` walks an object
    graph and is not worth paying for on every step of a run that is not
    compiling.

    ``miss=lo-hi`` numbers the cache misses this region is responsible for. The
    keys ``enable_miss_keys`` prints arrive on stderr in miss order within a
    process, so those two numbers name the exact lines to read rather than
    leaving them to be matched by interleaving, which the C++ stream and the
    Python logger do not reliably agree on.
    """
    global _last, _last_miss
    if not envs.TPU_GRAPH_PROBE:
        return
    total = total_entries()
    misses = total_misses()
    delta = total - _last
    miss_lo = _last_miss + 1
    _last, _last_miss = total, misses
    if delta and _keys_wanted:
        try:
            if not torch.tpu._get_dump_on_cache_miss():
                logger.warning(
                    "[GRAPH_PROBE] dump-on-cache-miss was off at "
                    "%s; re-asserted -> %s", tag, _assert_miss_keys())
        except Exception as e:  # noqa: BLE001
            logger.warning("[GRAPH_PROBE] miss-key flag unreadable: %s", e)
    if delta:
        extra = " ".join(f"{k}={_field(v)}" for k, v in fields.items())
        logger.info("[GRAPH_PROBE] %s new=%d total=%d miss=%d-%d %s", tag,
                    delta, total, miss_lo, misses, extra)


def probe_boundary(tag: str, **fields) -> None:
    """Probe with the lazy graph drained first, for an exact region boundary.

    Ops traced before an ordinary ``probe`` are only counted there if something
    already materialized them, so consecutive regions that never synchronize
    all report their programs against whichever one syncs first. Draining costs
    a device round trip, which is why it is not what ``probe`` does.

    It also costs accuracy of a different kind. Under DEFER_AND_FUSE the
    program is built at the sync, so a drain does not merely observe the
    regions -- it decides where the fusion boundaries fall, and a drained run
    compiles a different set of programs than production does (41 entries per
    rank drained against 26 undrained, on the same workload). Set
    ``TPU_GRAPH_PROBE_DRAIN=0`` to keep production's fusion and accept that a
    region's count then belongs to it and its undrained predecessors together.
    """
    if not envs.TPU_GRAPH_PROBE:
        return
    if envs.TPU_GRAPH_PROBE_DRAIN:
        try:
            synchronize_tensors(wait=True)
        except Exception as e:
            logger.warning("[GRAPH_PROBE] could not drain before %s: %s", tag,
                           e)
    probe(tag, **fields)


_dumped: dict[str, int] = {}

# Enough repeats of one tag to show whether a context that compiles twice
# builds the same graph both times, and few enough that a whole run's dumps
# stay readable.
_MAX_DUMPS_PER_TAG = 4


def capture_graph(*tensors) -> tuple[str, int] | None:
    """The MLIR of the deferred graph producing ``tensors``, and the cache size.

    Pass several tensors when the sync being mirrored flushes several at once
    -- the graph is the one program that produces them all.

    A cache entry exposes ``compilation_duration``, ``read_count`` and
    ``last_read`` and nothing else, so the counters can report that a region
    compiled a second program at shapes the warmup already swept without
    saying how the two differ. The graph text is that missing identity.

    Call this immediately before the sync that builds the graph: afterwards
    the tensor is a materialized buffer with nothing left to traverse. Pair it
    with ``write_graph`` after the sync, which is what knows whether the graph
    was one the cache already had.
    """
    if not envs.TPU_GRAPH_PROBE_MLIR_DIR:
        return None
    try:
        from torch_tpu._internal import sync
        return sync.computation_mlir(*tensors), total_entries()
    except Exception as e:  # noqa: BLE001 - diagnostics must not break a run
        logger.warning("[GRAPH_PROBE] could not read a graph: %s", e)
        return None


def write_graph(captured: tuple[str, int] | None, tag: str) -> None:
    """Write a captured graph: the first per tag, then only when it compiles.

    Dumping every occurrence would bury the interesting ones: a served pair
    recurs across dozens of requests and compiles on one of them. Keeping the
    first of each tag gives the diff a baseline even for a tag whose program
    the cache already had, and keeping the later compiling ones is what shows
    a pair building a second, different program.
    """
    directory = envs.TPU_GRAPH_PROBE_MLIR_DIR
    if captured is None or not directory:
        return
    text, entries_before = captured
    seq = _dumped.get(tag, 0)
    if seq and total_entries() == entries_before:
        return
    if seq >= _MAX_DUMPS_PER_TAG:
        return
    _dumped[tag] = seq + 1
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{tag}.{seq}.{os.getpid()}.mlir")
        with open(path, "w") as f:
            f.write(text)
        logger.info("[GRAPH_PROBE] graph -> %s", path)
    except Exception as e:  # noqa: BLE001 - diagnostics must not break a run
        logger.warning("[GRAPH_PROBE] could not write %s: %s", tag, e)


def reset() -> None:
    """Re-baseline, so a following probe reports only what comes after it."""
    global _last, _last_miss
    if envs.TPU_GRAPH_PROBE:
        _last = total_entries()
        _last_miss = total_misses()
