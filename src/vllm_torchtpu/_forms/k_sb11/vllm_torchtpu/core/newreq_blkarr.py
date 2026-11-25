# SPDX-License-Identifier: Apache-2.0
"""Ships a new request's block ids to the workers as one int32 buffer (VLLM_TPU_NEWREQ_BLKARR; q38 SV-NEWREQ2 S1).

A ``NewRequestData`` carries the request's block table: one Python int list per KV cache group (59 groups for Qwen4Exp
decode, about 10k ids for a 104k-token prompt). The engine core pickles every int into the step's ``SchedulerOutput``,
each worker unpickles them back into int objects, and ``InputBatch.add_request`` converts them to int32 one element at a
time when ``BlockTable.append_row`` assigns the row (``self.block_table.np[row, s:e] = block_ids``).

With the key on, the engine core's ``NewRequestData.from_request`` hands out the scheduler's block ids in a
``PackedBlockIds``: a tuple subclass holding the scheduler's own lists, so every engine-side reader (the KV connectors,
the scheduler) sees exactly the stock object contents. Only pickling differs: ``PackedBlockIds.__reduce_ex__`` packs
the live lists at that moment (in the engine core, inside ``MessageQueue.enqueue``) into one int32 buffer plus the
per-group lengths. Because the pack is taken from the lists at pickle time, what ships always equals what the stock
pickle of those lists would carry; there is no earlier copy that could go stale. On the worker the reconstructor
returns a tuple of per-group ``array.array('i')``:

- ``BlockTable.append_row`` assigns it through the buffer protocol (one memcpy per group) instead of an element-wise
  int conversion; ``if not block_ids`` and ``len(block_ids)`` behave as for a list;
- ``GPUModelRunner._update_states`` extends a running request's groups with the scheduler's new lists
  (``block_ids.extend(new_ids)``, gpu_model_runner.py:1441-1442), which ``array.array`` supports;
- vLLM's BlockTable mutators are untouched: a packed admission still goes through ``MultiGroupBlockTable.add_row`` ->
  ``BlockTable.add_row`` -> ``append_row``, so PAM-INCR's dirty-row marks and attention_metadata's version bump (both
  wrappers read only ``len(block_ids)``) see every row it writes.

Fallbacks (the stock tuple of lists ships): an id outside int32 or a non-int id, a request with no block id at all, a
block table that is not a tuple of lists, and every request in an A/B OFF window.

``VLLM_TPU_NEWREQ_BLKARR=1`` turns it on (default off: nothing is patched). ``VLLM_TPU_NEWREQ_BLKARR_AB_MS=<P>``
(0 = off): in-process A/B in wall-clock windows, packed ids in even ``time_ns // P ms`` windows and lists in odd ones;
P must divide 86,400,000 with an even quotient. Both keys are host-only and excluded from the compile key.

Install points: the engine core patches ``NewRequestData.from_request`` (patch registry, engine_core stage, after
prompt_token_array); each worker's ``TPUModelRunner.__init__`` wraps ``InputBatch.add_request`` to count and time
admissions by container type (it calls vLLM's own function unchanged). Both refuse loudly (RuntimeError) if a vLLM
function they patch or rely on differs from the source this copy was written against.

Runtime stats lines (from the threads that do the work):

- engine core main thread (it schedules and pickles), every LOG_EVERY ``from_request`` calls through the patch:
  ``NEWREQ_BLKARR side=engine calls=.. wrapped=.. packed=.. bytes=.. fallback=.. overflow=.. empty=.. type=..
  ab_off=.. ids_mean=.. pack_us_mean=..`` (packed: buffers built at pickle time; bytes: int32 bytes shipped;
  pack_us_mean: engine-side cost per packed pickle, the pack plus ``tobytes``);
- each worker, every LOG_EVERY ``add_request`` calls: ``NEWREQ_BLKARR side=worker adds=.. packed_adds=..
  list_adds=.. packed_add_us_mean=.. list_add_us_mean=.. unpacked=.. unpacked_ids_mean=.. rank=..``
  (packed_adds: admissions whose block ids are int32 arrays, including re-adds of such requests).

An A/B window line ``NEWREQ_BLKARR AB window=.. on=..`` is logged by the engine core at each window change.
"""

import array
import copy
import functools
import hashlib
import inspect
import time
import weakref

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

DAY_MS = 86_400_000
# About 0.19 admissions a decode step at B1 (about 8 a second per engine): an engine line about once a minute; the
# worker line counts re-adds too.
LOG_EVERY = 500
_PATCH_FLAG = "_tpu_newreq_blkarr_patch"
# sha256 of the normalized sources (blank lines dropped, lines right-stripped, "\0" after each function) this copy was
# written against, vLLM 0.31: NewRequestData.from_request (patched, engine core); InputBatch.add_request (wrapped,
# worker); the worker-side readers of the packed ids: BlockTable.append_row, BlockTable.add_row,
# MultiGroupBlockTable.append_row, MultiGroupBlockTable.add_row, GPUModelRunner._update_states.
FROM_REQUEST_SHA256 = "6dc7a2730e7945229567b578c34ad56972600f936d1e960d957a2a179b90489e"
ADD_REQUEST_SHA256 = "9f2851de055dfe0de53c9e4a29350ca4f8f618377b90b09c9855cf79a5f88db4"
WORKER_DEPS_SHA256 = "76e92239d4bbc14c1736176f6a1b66fdfc33915a8300614a81500b6be790f49a"

ENABLED = False
AB_MS = 0
_state = {"ab_window": None, "runner": None}


def _new_stats() -> dict:
    return {"calls": 0, "wrapped": 0, "packed": 0, "ids": 0, "bytes": 0, "pack_ns": 0, "fallback": 0, "overflow": 0,
            "empty": 0, "type": 0, "ab_off": 0}


def _new_wstats() -> dict:
    return {"adds": 0, "packed_adds": 0, "list_adds": 0, "packed_ns": 0, "list_ns": 0, "unpacked": 0,
            "unpacked_ids": 0}


stats = _new_stats()
wstats = _new_wstats()


def _load() -> None:
    global ENABLED, AB_MS
    ENABLED = envs.VLLM_TPU_NEWREQ_BLKARR
    AB_MS = envs.VLLM_TPU_NEWREQ_BLKARR_AB_MS
    if AB_MS and (DAY_MS % AB_MS or (DAY_MS // AB_MS) % 2):
        raise ValueError(
            f"Invalid value '{AB_MS}' for VLLM_TPU_NEWREQ_BLKARR_AB_MS: must be 0 or divide {DAY_MS} with an even "
            "quotient."
        )


_load()


def enabled() -> bool:
    return bool(ENABLED)


def _now_ns() -> int:
    return time.time_ns()


def ab_on() -> bool:
    """Whether packed ids are on in the current A/B window (always, without AB)."""
    if AB_MS <= 0:
        return True
    k = _now_ns() // (AB_MS * 1_000_000)
    if k != _state["ab_window"]:
        _state["ab_window"] = k
        logger.info(
            "NEWREQ_BLKARR AB window=%d on=%d calls=%d packed=%d fallback=%d ab_off=%d",
            k, int(k % 2 == 0), stats["calls"], stats["packed"], stats["fallback"], stats["ab_off"],
        )
    return k % 2 == 0


def source_sha256(*fns) -> str:
    h = hashlib.sha256()
    for fn in fns:
        src = inspect.getsource(inspect.unwrap(fn))
        h.update("\n".join(line.rstrip() for line in src.splitlines() if line.strip()).encode())
        h.update(b"\0")
    return h.hexdigest()


def unpack_block_ids(buf: bytes, lens: tuple) -> tuple:
    """The reconstructor (worker side): one int32 array per KV cache group, in group order."""
    a = array.array("i")
    a.frombytes(buf)
    out = []
    off = 0
    for n in lens:
        out.append(a[off : off + n])
        off += n
    w = wstats
    w["unpacked"] += 1
    w["unpacked_ids"] += len(a)
    return tuple(out)


def _pack(groups):
    """(int32 buffer, per-group lengths) of the live lists, or (None, reason)."""
    buf = array.array("i")
    try:
        for ids in groups:
            buf.fromlist(ids)
    except OverflowError:
        return None, "overflow"
    except TypeError:
        return None, "type"
    if not buf:
        return None, "empty"
    return buf, tuple(len(ids) for ids in groups)


class PackedBlockIds(tuple):
    """A new request's block ids on the engine core: the scheduler's own lists. Pickling ships them as one int32
    buffer, packed from the lists as they are at that moment."""

    __slots__ = ()

    def __reduce_ex__(self, protocol):
        s = stats
        t0 = time.perf_counter_ns()
        buf, lens = _pack(self)
        if buf is None:
            s["fallback"] += 1
            s[lens] += 1
            return (tuple, (list(self),))
        out = (unpack_block_ids, (buf.tobytes(), lens))
        s["pack_ns"] += time.perf_counter_ns() - t0
        s["packed"] += 1
        s["ids"] += len(buf)
        s["bytes"] += len(out[1][0])
        return out

    def __reduce__(self):
        return self.__reduce_ex__(2)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return PackedBlockIds(copy.deepcopy(ids, memo) for ids in self)


def _tick() -> None:
    s = stats
    if s["calls"] % LOG_EVERY == 0:
        # The engine-side runtime stats line (engine core main thread, which schedules and pickles).
        logger.info(
            "NEWREQ_BLKARR side=engine calls=%d wrapped=%d packed=%d bytes=%d fallback=%d overflow=%d empty=%d "
            "type=%d ab_off=%d ids_mean=%.0f pack_us_mean=%.1f",
            s["calls"], s["wrapped"], s["packed"], s["bytes"], s["fallback"], s["overflow"], s["empty"], s["type"],
            s["ab_off"], s["ids"] / max(1, s["packed"]), s["pack_ns"] / 1e3 / max(1, s["packed"]),
        )


def new_request_data(data):
    """Hands ``data``'s block ids out as a ``PackedBlockIds`` (the same lists) when they can ship packed."""
    s = stats
    s["calls"] += 1
    if not ab_on():
        s["ab_off"] += 1
    else:
        ids = data.block_ids
        if type(ids) is not tuple or not all(type(g) is list for g in ids):
            s["fallback"] += 1
            s["type"] += 1
        elif not any(ids):
            s["fallback"] += 1
            s["empty"] += 1
        else:
            data.block_ids = PackedBlockIds(ids)
            s["wrapped"] += 1
    _tick()
    return data


def patch_newreq_blkarr():
    """engine_core stage; a no-op (nothing imported, nothing patched) unless VLLM_TPU_NEWREQ_BLKARR is set."""
    if not enabled():
        return None
    from vllm.v1.core.sched.output import NewRequestData

    cur = NewRequestData.__dict__["from_request"].__func__
    if getattr(cur, _PATCH_FLAG, False):
        return True
    sha = source_sha256(cur)
    if sha != FROM_REQUEST_SHA256 or array.array("i").itemsize != 4:
        raise RuntimeError(
            "VLLM_TPU_NEWREQ_BLKARR refused: vLLM's NewRequestData.from_request (%s, want %s) differs from the source "
            "the packed block ids were written against, or array('i') is not 4 bytes (%d)"
            % (sha[:12], FROM_REQUEST_SHA256[:12], array.array("i").itemsize)
        )

    def from_request(cls, request, *args, **kwargs):
        return new_request_data(cur(cls, request, *args, **kwargs))

    setattr(from_request, _PATCH_FLAG, True)
    from_request.__wrapped__ = cur
    NewRequestData.from_request = classmethod(from_request)
    logger.info(
        "VLLM_TPU_NEWREQ_BLKARR=1: new requests' block ids ship as one int32 buffer (AB %d ms; engine stats every "
        "%d new requests)",
        AB_MS, LOG_EVERY,
    )
    return True


def _rank():
    ref = _state["runner"]
    runner = ref() if ref is not None else None
    return getattr(runner, "_profiler_rank", None)


def make_add_request(stock):
    """``InputBatch.add_request`` counted and timed by the container type of the request's block ids."""

    @functools.wraps(stock)
    def add_request(self, request):
        ids = request.block_ids
        packed = bool(ids) and type(ids[0]) is array.array
        t0 = time.perf_counter_ns()
        out = stock(self, request)
        dt = time.perf_counter_ns() - t0
        w = wstats
        w["adds"] += 1
        if packed:
            w["packed_adds"] += 1
            w["packed_ns"] += dt
        else:
            w["list_adds"] += 1
            w["list_ns"] += dt
        if w["adds"] % LOG_EVERY == 0:
            # The worker-side runtime stats line (the model thread, which adds the requests).
            logger.info(
                "NEWREQ_BLKARR side=worker adds=%d packed_adds=%d list_adds=%d packed_add_us_mean=%.1f "
                "list_add_us_mean=%.1f unpacked=%d unpacked_ids_mean=%.0f rank=%s",
                w["adds"], w["packed_adds"], w["list_adds"], w["packed_ns"] / 1e3 / max(1, w["packed_adds"]),
                w["list_ns"] / 1e3 / max(1, w["list_adds"]), w["unpacked"],
                w["unpacked_ids"] / max(1, w["unpacked"]), _rank(),
            )
        return out

    setattr(add_request, _PATCH_FLAG, True)
    return add_request


def patch_worker(runner=None) -> bool:
    """TPUModelRunner.__init__; a no-op (nothing imported, nothing patched) unless VLLM_TPU_NEWREQ_BLKARR is set."""
    if not enabled():
        return False
    from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if runner is not None:
        try:
            _state["runner"] = weakref.ref(runner)
        except TypeError:
            _state["runner"] = lambda: runner
    cur = InputBatch.add_request
    if getattr(cur, _PATCH_FLAG, False):
        return True
    asha = source_sha256(cur)
    dsha = source_sha256(
        BlockTable.append_row, BlockTable.add_row, MultiGroupBlockTable.append_row, MultiGroupBlockTable.add_row,
        GPUModelRunner._update_states,
    )
    if asha != ADD_REQUEST_SHA256 or dsha != WORKER_DEPS_SHA256:
        raise RuntimeError(
            "VLLM_TPU_NEWREQ_BLKARR refused: vLLM's InputBatch.add_request (%s, want %s) or the readers of the packed "
            "block ids (BlockTable / MultiGroupBlockTable append_row / add_row, GPUModelRunner._update_states: %s, "
            "want %s) differ from the sources this copy was written against"
            % (asha[:12], ADD_REQUEST_SHA256[:12], dsha[:12], WORKER_DEPS_SHA256[:12])
        )
    InputBatch.add_request = make_add_request(cur)
    logger.info(
        "VLLM_TPU_NEWREQ_BLKARR=1: worker admissions counted by block-id container (worker stats every %d "
        "adds)",
        LOG_EVERY,
    )
    return True
