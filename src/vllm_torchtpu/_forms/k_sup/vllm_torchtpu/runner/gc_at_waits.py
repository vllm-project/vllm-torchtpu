# SPDX-License-Identifier: Apache-2.0
"""Runs due garbage collections while the main thread waits (VLLM_TPU_GC_AT_WAITS).

CPython collects generation 0 when its allocation count passes the threshold
(700), wherever that allocation happens. In the decode, that is mostly on the
critical path between a step's output and the next step's forward: each young
96k-token request list makes a generation-0 pass cost about 1-3 ms, and every
tenth pass is a generation-1 pass of about 20 ms. Both processes on that path
also wait every step: the engine core for the worker's output, a worker for
the next SchedulerOutput and for the previous step's sampled-token copy.

At each of those waits, when the generation-0 count has reached half the
threshold, this runs the collection CPython would run next (generation 1 when
its count is one short of its threshold, else generation 0). The waits keep
the count below the threshold, so automatic collections become rare; they
stay enabled, and generation 2 stays with CPython's own policy, which the
generation-1 passes here advance as usual. Only the main thread collects: a
collection on another thread holds the GIL against the main thread.

Waits: ``SpinCondition.wait`` (a shared-memory message queue reader with no
message ready; engine core and workers, patched here) and, on workers,
``AsyncTPUCopyState._wait`` when its ready event has not fired (it calls
``collect_due`` itself).

Generation 2 in the engine core. A generation-2 pass traverses every tracked,
unfrozen object and every reference those objects hold. In the decode engine
core most of those references are the token lists of the requests it holds
(two lists of about 96k ints per request), so the pass grows with the requests
held. Workers freeze their heap after warmup and run no generation-2 pass.
Each part below has its own flag, default off:

- ``VLLM_TPU_GC_FREEZE``: a full collection and ``gc.freeze()`` once
  ``EngineCoreProc`` is built, which is after warmup and KV-cache init and
  before the busy loop. Upstream ``EngineCore.__init__`` freezes before
  ``EngineCoreProc`` and the TPU patches finish their setup; this also freezes
  what they add.
- ``VLLM_TPU_GC_UNTRACK_TOKEN_LISTS``: a Request's ``prompt_token_ids`` and
  ``_all_token_ids`` leave the collector when the Request is built. They hold
  only ints, which the collector does not track, so they cannot be part of a
  reference cycle; CPython untracks tuples of atomic objects for the same
  reason. Reference counting frees them as before.
- ``VLLM_TPU_GC_GEN2_THRESHOLD``: generation 2's threshold, set at the freeze
  point, so automatic generation-2 passes become rare.
- ``VLLM_TPU_GC_GEN2_PERIOD_S``: once the period has passed since the last
  generation-2 pass (automatic or not), a generation-2 pass at the first
  message-queue wait that has lasted 2 ms, or at any wait once two periods
  have passed. The engine core makes about 16 waits a step (a read from each
  rank after execute_model and after sample_tokens) and one of them lasts
  most of the device step; a wait that reaches 2 ms is that one. It installs
  the ``SpinCondition`` hooks without ``VLLM_TPU_GC_AT_WAITS``; generations 0
  and 1 then stay with CPython.
- ``VLLM_TPU_GC_CENSUS``: held-request counts such as ``0:60:90:120`` (colons
  or commas). At the
  freeze point for 0, and at the first wait at or above each other count, a
  census of the tracked heap (type table, owners of large containers) and the
  length of one generation-2 pass go to the log as ``GC_CENSUS`` lines. Every
  generation-2 pass then logs a ``GC_GEN2`` line: its length, cause, objects
  collected and the process's resident memory.
"""

import gc
import re
import sys
import threading
import time
import types
from collections import deque

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENABLED = envs.VLLM_TPU_GC_AT_WAITS
FREEZE = envs.VLLM_TPU_GC_FREEZE
UNTRACK_TOKEN_LISTS = envs.VLLM_TPU_GC_UNTRACK_TOKEN_LISTS
GEN2_THRESHOLD = envs.VLLM_TPU_GC_GEN2_THRESHOLD
GEN2_PERIOD_S = envs.VLLM_TPU_GC_GEN2_PERIOD_S
CENSUS = envs.VLLM_TPU_GC_CENSUS
_PATCH_FLAG = "_tpu_gc_at_waits_patch"
_MAIN = threading.main_thread()
# Collections run here, by generation.
counts = [0, 0, 0]
# A container this long is listed with its owner in the census.
_BIG = 4096
# Age a wait reaches before a scheduled generation-2 pass runs in it. At the
# decode's cap band (60-64 running, j-dee2086a), 90% of the engine core's
# waits that last 2 ms go on for at least 8 ms more (median 20 ms).
_GEN2_AFTER_WAIT_NS = 2_000_000
# SpinCondition attribute: when the current wait's first wait() call ran.
_WAIT_START = "_tpu_gc_wait_start_ns"
_CONTAINERS = (list, tuple, set, frozenset, deque)


class _EngineGC:
    """Engine-core generation-2 state; stays empty in workers."""

    def __init__(self) -> None:
        self.started = False
        self.engine = None
        self.period_ns = 0
        # End of the last generation-2 pass, automatic or not.
        self.last_ns = 0
        self.start_ns = 0
        self.cause = "automatic"
        self.census_at: list[int] = []
        self.log_gen2 = False


_G = _EngineGC()


def collect_due() -> None:
    """Runs the next due collection when called on the main thread."""
    c0, c1, _ = gc.get_count()
    t0, t1, _ = gc.get_threshold()
    if t0 == 0 or c0 < t0 // 2 or threading.current_thread() is not _MAIN:
        return
    generation = 1 if c1 + 1 >= t1 else 0
    gc.collect(generation)
    counts[generation] += 1


def at_wait(cond) -> None:
    """Runs what is due at a message-queue wait: a census, generation 2, or younger.

    cond is the waiting SpinCondition.
    """
    if threading.current_thread() is not _MAIN:
        return
    if _G.census_at and held_requests() >= _G.census_at[0]:
        _log_census()
        return
    if _G.period_ns:
        now = time.monotonic_ns()
        since = now - _G.last_ns
        if since >= _G.period_ns:
            start = getattr(cond, _WAIT_START, 0)
            if not start:
                start = now
                setattr(cond, _WAIT_START, now)
            if now - start >= _GEN2_AFTER_WAIT_NS or since >= 2 * _G.period_ns:
                _collect_gen2("scheduled")
                return
    if ENABLED:
        collect_due()


def _log_census() -> None:
    held = held_requests()
    mark = _G.census_at.pop(0)
    while _G.census_at and held >= _G.census_at[0]:
        mark = _G.census_at.pop(0)
    try:
        lines = census(f"held>={mark}")
    except Exception as exc:  # a diagnostic must not stop the engine core
        logger.warning("GC_CENSUS held>=%d failed: %r", mark, exc)
        return
    for line in lines:
        logger.info("GC_CENSUS %s", line)


def _collect_gen2(cause: str) -> int:
    _G.cause = cause
    try:
        collected = gc.collect(2)
    finally:
        _G.cause = "automatic"
    counts[2] += 1
    _G.last_ns = time.monotonic_ns()
    return collected


def held_requests() -> int:
    """Requests the engine core's scheduler holds; 0 outside the engine core."""
    if _G.engine is None:
        return 0
    return len(_G.engine.scheduler.requests)


def event_pending(event) -> bool:
    """Whether a torch.tpu.Event has not completed; False when unknown."""
    try:
        return not event.query()
    except Exception:
        return False


def patch_gc_at_waits() -> None:
    if not (ENABLED or GEN2_PERIOD_S > 0 or CENSUS):
        return
    from vllm.distributed.device_communicators.shm_broadcast import SpinCondition

    if getattr(SpinCondition.wait, _PATCH_FLAG, False):
        return
    orig_wait = SpinCondition.wait
    orig_record_read = SpinCondition.record_read

    def wait(self, timeout_ms=None):
        at_wait(self)
        return orig_wait(self, timeout_ms)

    # A read ends the wait: the next wait() call starts a new one.
    def record_read(self):
        setattr(self, _WAIT_START, 0)
        return orig_record_read(self)

    for fn, orig in ((wait, orig_wait), (record_read, orig_record_read)):
        setattr(fn, _PATCH_FLAG, True)
        fn.__wrapped__ = orig
    SpinCondition.wait = wait
    if GEN2_PERIOD_S > 0:
        SpinCondition.record_read = record_read
    logger.info(
        "VLLM_TPU_GC_AT_WAITS=%d: due collections run at message-queue waits",
        int(ENABLED),
    )


# --- engine core: generation 2 ----------------------------------------------


def patch_engine_core_gc() -> None:
    """Installs the engine core's generation-2 parts (engine_core stage)."""
    if not (
        FREEZE or UNTRACK_TOKEN_LISTS or GEN2_THRESHOLD or GEN2_PERIOD_S > 0 or CENSUS
    ):
        return
    if UNTRACK_TOKEN_LISTS:
        _patch_request_untrack()
    from vllm.v1.engine.core import EngineCoreProc

    init = EngineCoreProc.__init__
    if getattr(init, _PATCH_FLAG, False):
        return

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        start_engine_gc(self)

    setattr(__init__, _PATCH_FLAG, True)
    __init__.__wrapped__ = init
    EngineCoreProc.__init__ = __init__


def start_engine_gc(engine) -> None:
    """Freezes the startup heap and sets the generation-2 policy, once."""
    if _G.started:
        return
    _G.started = True
    _G.engine = engine
    if FREEZE:
        start = time.perf_counter()
        gc.collect()
        before = gc.get_freeze_count()
        gc.freeze()
        logger.info(
            "VLLM_TPU_GC_FREEZE: froze %d objects (%d frozen in all) in %.0f ms",
            gc.get_freeze_count() - before,
            gc.get_freeze_count(),
            (time.perf_counter() - start) * 1e3,
        )
    if GEN2_THRESHOLD:
        t0, t1, _ = gc.get_threshold()
        gc.set_threshold(t0, t1, GEN2_THRESHOLD)
    if GEN2_PERIOD_S > 0:
        _G.period_ns = int(GEN2_PERIOD_S * 1e9)
        _G.last_ns = time.monotonic_ns()
    if CENSUS:
        _G.census_at = sorted(int(x) for x in re.split("[,:]", CENSUS) if x.strip())
        _G.log_gen2 = True
    if _G.period_ns or _G.log_gen2:
        gc.callbacks.append(_gen2_callback)
    logger.info(
        "Engine core GC: freeze=%d untrack_token_lists=%d thresholds=%s "
        "gen2_period_s=%g census_at=%s",
        int(FREEZE),
        int(UNTRACK_TOKEN_LISTS),
        gc.get_threshold(),
        GEN2_PERIOD_S,
        _G.census_at,
    )
    if _G.census_at and _G.census_at[0] <= 0:
        _log_census()


def _gen2_callback(phase: str, info: dict) -> None:
    if info.get("generation") != 2:
        return
    now = time.monotonic_ns()
    if phase == "start":
        _G.start_ns = now
        return
    _G.last_ns = now
    if _G.log_gen2:
        logger.info(
            "GC_GEN2 %.1f ms cause=%s held=%d collected=%d rss_mb=%.0f",
            (now - _G.start_ns) / 1e6,
            _G.cause,
            held_requests(),
            info.get("collected", 0),
            _rss_mb(),
        )


_untrack_fn = None


def untrack_int_list(values) -> None:
    """Takes a list that holds only ints out of the collector."""
    global _untrack_fn
    if type(values) is not list or not gc.is_tracked(values):
        return
    if _untrack_fn is None:
        import ctypes

        fn = ctypes.pythonapi.PyObject_GC_UnTrack
        fn.argtypes = (ctypes.py_object,)
        fn.restype = None
        _untrack_fn = fn
    _untrack_fn(values)


def _patch_request_untrack() -> None:
    from vllm.v1.request import Request

    init = Request.__init__
    if getattr(init, _PATCH_FLAG, False):
        return

    # EngineCoreRequest.prompt_token_ids is list[int], which msgspec checks on
    # decode; _all_token_ids starts as its copy and only takes sampled ids.
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        untrack_int_list(self.prompt_token_ids)
        untrack_int_list(self._all_token_ids)

    setattr(__init__, _PATCH_FLAG, True)
    __init__.__wrapped__ = init
    Request.__init__ = __init__


# --- census -----------------------------------------------------------------


def _module(tp: type) -> str:
    # Some types (descriptor-backed metaclasses) answer __module__ with a
    # non-string.
    mod = getattr(tp, "__module__", "")
    return mod if isinstance(mod, str) else ""


def _type_name(tp: type) -> str:
    mod = _module(tp)
    return tp.__qualname__ if mod == "builtins" else f"{mod}.{tp.__qualname__}"


# Heap objects are classified by type(o), not isinstance(): isinstance falls
# back to o.__class__, which raises for a dead weakref proxy.
def _refs(o) -> int:
    tp = type(o)
    if issubclass(tp, dict):
        return 2 * len(o)
    if issubclass(tp, _CONTAINERS):
        return len(o)
    return 0


def _sizeof(o) -> int:
    try:
        return sys.getsizeof(o)
    except Exception:
        # A Python-level __sizeof__ can raise; the census counts 0 bytes.
        return 0


def _attr_of(owner, value) -> str:
    """Name of the attribute of owner that holds value; '?' when not found."""
    d = getattr(owner, "__dict__", None)
    if isinstance(d, dict):
        for key, v in d.items():
            if v is value:
                return str(key)
    for klass in type(owner).__mro__:
        slots = klass.__dict__.get("__slots__", ())
        for key in (slots,) if isinstance(slots, str) else slots:
            if getattr(owner, key, None) is value:
                return key
    return "?"


def _rss_mb() -> float:
    import resource

    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
    except (OSError, ValueError, IndexError):
        return -1.0
    return pages * resource.getpagesize() / 2**20


def census(label: str, top: int = 30, time_gen2: bool = True) -> list[str]:
    """Counts the tracked heap by type and names the owners of large containers.

    Returns log lines. "refs" is the references a collection traverses (the
    length of a list, tuple, set or deque, twice a dict's); "bytes" is
    sys.getsizeof. Frozen and untracked objects are not counted. With
    time_gen2, one generation-2 pass runs and its length is reported.
    """
    start = time.perf_counter()
    objs = gc.get_objects()
    table: dict[type, list[int]] = {}
    big: dict[int, object] = {}
    for o in objs:
        rec = table.get(type(o))
        if rec is None:
            rec = table[type(o)] = [0, 0, 0]
        n = _refs(o)
        rec[0] += 1
        rec[1] += n
        rec[2] += _sizeof(o)
        if n >= _BIG:
            big[id(o)] = o
    count_ms = (time.perf_counter() - start) * 1e3

    # Owners: vLLM and TPU objects that hold a large container directly.
    owners: dict[str, list[int]] = {}
    labelled: set[int] = set()
    for o in objs:
        tp = type(o)
        if not _module(tp).startswith("vllm") or issubclass(tp, (dict, *_CONTAINERS)):
            continue
        for r in gc.get_referents(o):
            # An instance's materialized __dict__ is a referent of its own.
            held = r.items() if type(r) is dict else (("", r),)
            for attr, v in held:
                if id(v) not in big or id(v) in labelled:
                    continue
                attr = attr or _attr_of(o, v)
                key = f"{tp.__qualname__}.{attr} ({type(v).__name__})"
                rec = owners.setdefault(key, [0, 0])
                rec[0] += 1
                rec[1] += _refs(v)
                labelled.add(id(v))
    # The rest, grouped by container and element type, with the referrers of
    # the largest container of each of the four largest groups.
    groups: dict[str, list] = {}
    for rid, o in big.items():
        if rid in labelled:
            continue
        first = next(iter(o.values() if issubclass(type(o), dict) else o), None)
        key = f"{type(o).__name__}[{type(first).__name__}]"
        g = groups.setdefault(key, [0, 0, 0, None])
        g[0] += 1
        g[1] += _refs(o)
        if _refs(o) > g[2]:
            g[2], g[3] = _refs(o), o
    reps = [(key, g[3]) for key, g in sorted(groups.items(), key=lambda kv: -kv[1][1])]
    for g in groups.values():
        g[3] = None
    objs = big = o = first = None
    exclude = {id(reps), *(id(rep) for rep in reps)}
    chains = {key: _referrer_chain(obj, exclude) for key, obj in reps[:4]}
    del reps
    owners_ms = (time.perf_counter() - start) * 1e3 - count_ms

    gen2_ms, collected = -1.0, -1
    if time_gen2:
        t = time.perf_counter()
        collected = _collect_gen2("census")
        gen2_ms = (time.perf_counter() - t) * 1e3

    tracked = sum(r[0] for r in table.values())
    refs = sum(r[1] for r in table.values())
    eng = _G.engine
    running = len(eng.scheduler.running) if eng is not None else 0
    waiting = len(eng.scheduler.waiting) if eng is not None else 0
    lines = [
        f"{label} held={held_requests()} running={running} waiting={waiting} "
        f"tracked={tracked} refs={refs} frozen={gc.get_freeze_count()} "
        f"thresholds={gc.get_threshold()} rss_mb={_rss_mb():.0f} "
        f"count_ms={count_ms:.0f} owners_ms={owners_ms:.0f} "
        f"gen2_ms={gen2_ms:.1f} gen2_collected={collected}"
    ]
    rows = [(_type_name(tp), *rec) for tp, rec in table.items()]
    for name, col in (("count", 1), ("refs", 2), ("bytes", 3)):
        for row in sorted(rows, key=lambda r: -r[col])[:top]:
            lines.append(
                f"{label} by {name}: {row[0]} count={row[1]} refs={row[2]} "
                f"bytes={row[3]}"
            )
    for key, (n, r) in sorted(owners.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"{label} owner: {key} containers={n} refs={r}")
    for key, g in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        lines.append(
            f"{label} unowned: {key} containers={g[0]} refs={g[1]} "
            f"referrers={chains.get(key, '-')}"
        )
    return lines


def _referrer_chain(obj, exclude: set[int], depth: int = 2) -> str:
    """Types of up to three tracked referrers of obj, and theirs for dicts."""
    found = gc.get_referrers(obj)
    exclude = exclude | {id(found)}
    out = []
    for r in found:
        if id(r) in exclude or type(r) is types.FrameType:
            continue
        name = type(r).__qualname__
        if depth > 1 and (
            issubclass(type(r), dict) or not _module(type(r)).startswith("vllm")
        ):
            if issubclass(type(r), dict):
                key = next((k for k, v in r.items() if v is obj), "?")
                name = f"dict[{str(key)[:40]}]"
            name = f"{name}<-({_referrer_chain(r, exclude, depth - 1)})"
        out.append(name)
        if len(out) == 3:
            break
    del found
    return "|".join(out) or "none tracked"
