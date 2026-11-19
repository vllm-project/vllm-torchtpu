# SPDX-License-Identifier: Apache-2.0
"""Prefix-affine data-parallel routing for a DP decode server (O4: DP4 x TP2).

Opt-in with ``VLLM_TPU_DP_AFFINITY=1``; host-only, outputs unchanged.

Why: each DP engine keeps its own KV pool and prefix cache. vLLM's DP client
(``DPLBAsyncMPClient.get_core_engine_for_request``) sends every request to the
least-loaded engine, so a conversation's next turn lands on the engine that
holds its previous turn's KV only about 1 time in ``num_engines``; the rest
reload the whole context over P->D and evict another conversation's prefix.

How: the router hashes the prompt in fixed ``chunk`` token chunks, chained so a
chunk's hash covers the whole prefix up to it, and keeps an LRU map from chunk
hash to the engine that last served a prompt with that prefix. A request goes
to the engine holding its **longest** matching prefix (the same conversation's
previous turn beats a prefix it only shares with other conversations of its
trace), unless that engine is overloaded: its stock load score exceeds
``(1 + slack) x mean + 1`` over the engines (bounded-load affinity), or, with
the optional slot guard (``VLLM_TPU_DP_AFFINITY_SLOT_GUARD=1``), its score is at
least max_num_seqs while another engine's is below. Then, and for prompts with
no match, vLLM's own least-loaded choice decides. The guard is off by default:
the stock score also counts requests still waiting for their P->D load, which
hold no running slot, so the guard refuses engines that still have free slots. Every routed
prompt's chunk hashes are recorded against the engine it went to.

The pure router below has no vLLM dependency (unit-tested by path); the patch
wraps the DP client and routes by setting ``request.data_parallel_rank``,
which the stock code honours.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from collections.abc import Sequence
from functools import wraps

_LOGGER_NAME = "vllm.vllm_torchtpu.core.dp_affinity"
_FLAG = "_tpu_dp_affinity"
_LOG_EVERY_S = 60.0
STATS = {"affine": 0, "fallback": 0, "new": 0, "pinned": 0, "errors": 0, "logged_at": 0.0}


def _logger():
    try:
        from vllm.logger import init_logger

        return init_logger(_LOGGER_NAME)
    except ImportError:
        return logging.getLogger(_LOGGER_NAME)


_SHM_MAGIC = 0x51385F4450414646  # "Q8_DPAFF"
_SHM_HEADER = 64
_KEY_MASK = (1 << 63) - 1


def _shm_key(h: int) -> int:
    """A chunk hash as a non-zero int64 (0 marks an empty slot)."""
    return (h & _KEY_MASK) or 1


class SharedPrefixTable:
    """Chunk hash -> engine map in POSIX shared memory, shared by every API-server
    process of one decode server (``VLLM_TPU_DP_AFFINITY_SHARED=1``).

    vLLM runs ``api_server_count`` = the DP size by default (4 on O4), each process
    with its own DP client, so per-process tables split one conversation's turns
    over 4 tables. This table is the same in all of them.

    Layout: a 64-byte header (magic, slots), then ``slots`` int64 keys, then
    ``slots`` int8 values (engine + 1; 0 = empty). Open addressing with ``probe``
    linear slots per key; when all are taken by other keys the home slot is
    overwritten (bounded memory, no eviction list). No locks: a writer stores the
    key, then the engine. A reader that races a writer can see a stale engine,
    which costs at most one prefix hit, never correctness (the bounded-load check
    and the stock fallback still apply). Chunk hashes are Python hashes of int
    tuples, which are identical across processes (no hash randomisation for ints).

    The segment name carries an epoch (the server's parent pid and its start
    time), so a restarted server never attaches to an older segment.
    """

    def __init__(self, name: str, slots: int = 1 << 21, probe: int = 8, wait_s: float = 5.0) -> None:
        import numpy as np
        from multiprocessing import shared_memory

        if slots < 1 or probe < 1:
            raise ValueError("bad SharedPrefixTable parameters")
        self.slots = 1 << max(10, (int(slots) - 1).bit_length())
        self.mask = self.slots - 1
        self.probe = min(int(probe), self.slots)
        self.name = name
        size = _SHM_HEADER + 9 * self.slots
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            self.created = True
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=name, create=False)
            self.created = False
        _untrack(shm)
        self._shm = shm
        head = np.ndarray((2,), dtype=np.int64, buffer=shm.buf, offset=0)
        if self.created:
            head[1] = self.slots
            head[0] = _SHM_MAGIC  # last: attachers wait for it
        else:
            deadline = time.monotonic() + wait_s
            while head[0] != _SHM_MAGIC:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"shared prefix table {name}: header never initialised")
                time.sleep(0.01)
            if int(head[1]) != self.slots or shm.size < size:
                raise RuntimeError(f"shared prefix table {name}: {int(head[1])} slots, expected {self.slots}")
        self.keys = np.ndarray((self.slots,), dtype=np.int64, buffer=shm.buf, offset=_SHM_HEADER)
        self.vals = np.ndarray((self.slots,), dtype=np.int8, buffer=shm.buf, offset=_SHM_HEADER + 8 * self.slots)

    def get(self, h: int) -> int | None:
        k = _shm_key(h)
        keys, vals, mask = self.keys, self.vals, self.mask
        i = k & mask
        for j in range(self.probe):
            s = (i + j) & mask
            kk = int(keys[s])
            if kk == k:
                v = int(vals[s])
                return v - 1 if v > 0 else None
            if kk == 0:
                return None
        return None

    def set(self, h: int, engine: int) -> None:
        k = _shm_key(h)
        keys, vals, mask = self.keys, self.vals, self.mask
        i = k & mask
        for j in range(self.probe):
            s = (i + j) & mask
            kk = int(keys[s])
            if kk == k:
                vals[s] = engine + 1
                return
            if kk == 0:
                keys[s] = k
                vals[s] = engine + 1
                return
        keys[i] = k
        vals[i] = engine + 1

    def __len__(self) -> int:
        import numpy as np

        return int(np.count_nonzero(self.keys))

    def close(self) -> None:
        self.keys = self.vals = None
        self._shm.close()


def _untrack(shm) -> None:
    """Keep Python's resource tracker from unlinking a segment other processes still use
    (Python < 3.13 registers attachers too and unlinks at their exit)."""
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(shm._name, "shared_memory")  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass


def _proc_start(pid: int) -> str:
    """Process start time in clock ticks since boot (/proc/<pid>/stat field 22), or '0'."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return "0"


def shared_table_name(pid: int | None = None) -> str:
    """Epoch-named segment for this decode server: its API servers share the parent pid."""
    override = os.environ.get("VLLM_TPU_DP_AFFINITY_SHARED_NAME")
    if override:
        return override
    pid = os.getppid() if pid is None else pid
    return f"q38dpaff_{pid}_{_proc_start(pid)}"


class PrefixAffinityRouter:
    """Longest-prefix engine lookup with bounded-load fallback (pure)."""

    def __init__(self, num_engines: int, chunk: int = 1024, slack: float = 0.25,
                 max_entries: int = 1_000_000, cap: int | None = None, table=None) -> None:
        if num_engines < 1 or chunk < 1 or slack < 0 or max_entries < 1 or (cap is not None and cap < 1):
            raise ValueError("bad PrefixAffinityRouter parameters")
        self.num_engines = num_engines
        self.chunk = chunk
        self.slack = slack
        self.max_entries = max_entries
        self.cap = cap
        self._map: OrderedDict[int, int] = OrderedDict()
        self.table = table  # SharedPrefixTable, or None for the per-process LRU

    def entries(self) -> int:
        return len(self.table) if self.table is not None else len(self._map)

    def chunk_hashes(self, token_ids: Sequence[int]) -> list[int]:
        """Chained hashes of the prompt's full ``chunk``-token chunks."""
        out, h = [], 0
        c = self.chunk
        for start in range(0, len(token_ids) - c + 1, c):
            h = hash((h, tuple(token_ids[start:start + c])))
            out.append(h)
        return out

    def lookup(self, hashes: Sequence[int]) -> tuple[int | None, int]:
        """(engine of the longest recorded prefix, its length in chunks)."""
        get = self.table.get if self.table is not None else self._map.get
        for i in range(len(hashes) - 1, -1, -1):
            engine = get(hashes[i])
            if engine is not None and engine < self.num_engines:
                return engine, i + 1
        return None, 0

    def within_bound(self, engine: int, scores: Sequence[float]) -> bool:
        """Bounded load: the target's score is at most ``(1 + slack) x mean + 1``, and
        (with ``cap``, the per-engine running slots) a full target is refused while
        another engine still has a free slot, so affinity never queues a request
        behind a full batch while a slot idles elsewhere."""
        mean = sum(scores) / len(scores)
        if scores[engine] > (1.0 + self.slack) * mean + 1.0:
            return False
        cap = self.cap
        return cap is None or scores[engine] < cap or min(scores) >= cap

    def record(self, hashes: Sequence[int], engine: int) -> None:
        if self.table is not None:
            for h in hashes:
                self.table.set(h, engine)
            return
        m = self._map
        for h in hashes:
            m[h] = engine
            m.move_to_end(h)
        while len(m) > self.max_entries:
            m.popitem(last=False)

    def choose(self, hashes: Sequence[int], scores: Sequence[float]) -> tuple[int | None, str]:
        """(engine, kind): kind 'affine' with an engine; 'fallback' / 'new' with None
        (the caller then takes its own least-loaded choice)."""
        engine, _ = self.lookup(hashes)
        if engine is None:
            return None, "new"
        if engine < len(scores) and self.within_bound(engine, scores):
            return engine, "affine"
        return None, "fallback"


def stock_score(client, idx: int) -> float:
    """``DPLBAsyncMPClient``'s own load score for engine ``idx`` (core_client.py:1605-1622)."""
    waiting, running, kv_cache_usage = client.lb_engines[idx]
    inflight = client.engine_inflight[client.core_engines[idx]]
    score = float(max(client.client_count * inflight, waiting + running))
    if waiting:
        score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
    return score


def route(client, request, orig, router: PrefixAffinityRouter, stats=STATS):
    """The patched ``get_core_engine_for_request`` body (testable with a fake client).

    Requests that already name a DP rank, carry no prompt tokens, or are pooling
    requests (late-interaction pooling has its own deterministic placement) take
    the stock path untouched."""
    if (request.data_parallel_rank is not None or not request.prompt_token_ids
            or getattr(request, "pooling_params", None) is not None):
        stats["pinned"] += request.data_parallel_rank is not None
        return orig(client, request)
    try:
        hashes = router.chunk_hashes(request.prompt_token_ids)
        n = len(client.core_engines)
        scores = [stock_score(client, i) for i in range(n)]
        engine, kind = router.choose(hashes, scores)
    except Exception:  # noqa: BLE001  never let a routing hint break serving
        stats["errors"] += 1
        return orig(client, request)
    if engine is not None:
        # The stock code honours a preset rank (and then records reqs_in_flight /
        # engine_inflight as usual); the rank is cleared again afterwards so the
        # request reaching the engine is exactly the stock one.
        request.data_parallel_rank = engine
        try:
            chosen = orig(client, request)
        finally:
            request.data_parallel_rank = None
        # The stock least-loaded branch counts its pick in the local snapshot
        # (between 100 ms coordinator updates); do the same for an affine pick.
        client.lb_engines[engine][0] += client.client_count
    else:
        chosen = orig(client, request)
        engine = client.core_engines.index(chosen)
    router.record(hashes, engine)
    stats[kind] += 1
    return chosen


def patch_dp_affinity() -> None:
    """Wrap ``DPLBAsyncMPClient.get_core_engine_for_request`` (API-server process)."""
    from vllm_torchtpu import envs

    if not envs.VLLM_TPU_DP_AFFINITY:
        return
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    if getattr(DPLBAsyncMPClient.get_core_engine_for_request, _FLAG, False):
        return
    orig = DPLBAsyncMPClient.get_core_engine_for_request
    chunk = int(envs.VLLM_TPU_DP_AFFINITY_CHUNK)
    slack = float(envs.VLLM_TPU_DP_AFFINITY_SLACK)
    max_entries = int(envs.VLLM_TPU_DP_AFFINITY_MAX_ENTRIES)
    slot_guard = bool(envs.VLLM_TPU_DP_AFFINITY_SLOT_GUARD)
    shared = bool(envs.VLLM_TPU_DP_AFFINITY_SHARED)
    shared_slots = int(envs.VLLM_TPU_DP_AFFINITY_SHARED_SLOTS)
    log = _logger()

    @wraps(orig)
    def get_core_engine_for_request(self, request):
        router = self.__dict__.get("_tpu_dp_router")
        if router is None:
            cap = None
            if slot_guard:
                cfg = getattr(self, "vllm_config", None)
                cap = getattr(getattr(cfg, "scheduler_config", None), "max_num_seqs", None)
            table = None
            if shared:
                name = shared_table_name()
                try:
                    table = SharedPrefixTable(name, shared_slots)
                    log.info("DP affinity shared table %s: %d slots, %s (pid %d).", name, table.slots,
                             "created" if table.created else "attached", os.getpid())
                except Exception as e:  # noqa: BLE001  never let the table break serving
                    STATS["errors"] += 1
                    log.warning("DP affinity shared table %s unavailable (%s); per-process table.", name, e)
            router = PrefixAffinityRouter(len(self.core_engines), chunk, slack, max_entries, cap, table)
            self.__dict__["_tpu_dp_router"] = router
            log.info("DP affinity router: %d engines, slot cap %s, table %s.", len(self.core_engines), cap,
                     "shared" if table is not None else "per-process")
        chosen = route(self, request, orig, router)
        now = time.monotonic()
        if now - STATS["logged_at"] >= _LOG_EVERY_S:
            STATS["logged_at"] = now
            log.info("DP affinity: affine=%d fallback=%d new=%d pinned=%d errors=%d entries=%d",
                     STATS["affine"], STATS["fallback"], STATS["new"], STATS["pinned"],
                     STATS["errors"], router.entries())
        return chosen

    setattr(get_core_engine_for_request, _FLAG, True)
    DPLBAsyncMPClient.get_core_engine_for_request = get_core_engine_for_request
    log.info("Applied TPU patch: prefix-affine DP routing (chunk %d tokens, slack %.2f, %d entries).",
             chunk, slack, max_entries)
