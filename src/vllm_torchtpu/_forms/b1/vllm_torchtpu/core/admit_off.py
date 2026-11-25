# SPDX-License-Identifier: Apache-2.0
"""Runs a decode's remote-KV admissions in the engine core's token wait (VLLM_TPU_ADMIT_OFF).

On a P/D decode (KV consumer) every new request starts with an async remote-KV load. vLLM admits it inside
``Scheduler.schedule()`` on the engine core's main thread, on the path from a step's tokens to the next dispatch: the
local prefix lookup, the connector's match, ``allocate_slots`` for the whole prompt (no tokens to compute) and the
connector's ``update_state_after_alloc``. At B1 decode rows that is about 2.35 ms per admission (4.6 ms while the input
thread builds a request), 0.67 ms per step on average (q38 SV-ADMIT, T64n probe, descriptive).

The main thread spends most of every step blocked, about 19 ms in the message-queue wait for the step's sampled
tokens. With the key on, once that wait has lasted ``OPEN_AFTER_NS``, each ``SpinCondition.wait()`` call on the main
thread does at most one unit of admission work:

* if the waiting queue is empty, it takes the input queue's head item when, and only when, it is an ADD (the order of
  the input queue is kept: a request is never taken past an abort or a utility call) and hands it to the engine's own
  ``_handle_client_request``, exactly as the busy loop would after the wait;
* if the head of the waiting queue is a fresh request whose remote KV loads asynchronously, it runs vLLM's own
  WAITING-loop admission for it (``admit_one``: the same calls with the same arguments, in the same order, as
  ``Scheduler.schedule``; the source fingerprints in ``_FINGERPRINTS`` must match) and moves it to the KV-holding queue
  in ``WAITING_FOR_REMOTE_KVS``, as the scan's skip path does. Its load is registered with the connector and goes out
  in the next ``schedule()``'s connector metadata, the same step as without the key.

At most ``VLLM_TPU_ADMIT_OFF`` requests (and at most the ``VLLM_TPU_KV_LOAD_STARTS_PER_STEP`` cap) are admitted
between two ``schedule()`` passes, and they are charged against the next pass's cap, so no step starts more loads than
the cap allows. Anything outside the narrow case (see ``ineligible``), and any admission whose load would not be async
or whose allocation fails, is left to ``schedule()`` with nothing changed (``admit_one`` backs out before its first
side effect). A ``finish_requests`` for a request admitted since the last ``schedule()`` (a client abort) is held and
re-issued right after the next ``schedule()`` returns, once its load is in the connector metadata, so the abort takes
the stock path of an abort during a load (without this, the consumer connector's ``request_finished`` would leave the
registered load in ``reqs_to_load``).

What moves: a load can be registered up to one pass earlier than without the key, the blocks it gets can differ (the
allocation runs before the previous step's ``update_from_output``), a prefix lookup in the wait still treats blocks
hashed in the previous pass as unwritten (so a local hit can only be shorter, and the remote load covers the rest), and
an abort of such a request is applied up to one pass later. Schedule-only: no device program, worker or model input
changes.

``VLLM_TPU_ADMIT_OFF=<N>`` turns it on (0 = off, the default: nothing is patched). ``VLLM_TPU_ADMIT_OFF_AB_MS=<P>``
(0 = off): in-process A/B in wall-clock windows (on in even ``time_ns // P ms`` windows; in odd windows nothing is done
in the wait). Both keys are host-only and excluded from the compile key. Engine-core process of a KV consumer only;
the engine core logs one ``ADMIT_OFF engaged:`` line once its engine is built (or ``ADMIT_OFF not engaged:`` with the
reason), and the main thread logs ``ADMIT_AHEAD ahead=<n> ... admit_ms_mean=<ms> unwritten_miss=<n> miss_tok=<t>
wait_tok=<T>`` at its first admission done in the wait and every ``LOG_EVERY`` admissions after (the runtime line).

``unwritten_miss`` / ``miss_tok`` / ``wait_tok`` size the one cache effect of the move (committed in-wait admissions
only, so ON windows only under AB): ``unwritten_miss`` counts admissions whose local hit stopped at a block that the
same-step patch still counts as unwritten (hashed in the pass just before the wait; ``schedule()`` would have hit it),
``miss_tok`` estimates the tokens such hits lost (walking forward while the hashes stay cached), and ``wait_tok`` sums
the prompt tokens of all committed in-wait admissions. Indicator, not an exact token count: the check reads the hashes of
the first full-attention KV cache group only (``unwritten_miss_tokens``), not every group's convergent hit.
"""

import ast
import hashlib
import inspect
import sys
import textwrap
import threading
import time

from vllm_torchtpu import envs
from vllm_torchtpu.core import kv_load_pacing
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LIMIT = envs.VLLM_TPU_ADMIT_OFF
AB_MS = envs.VLLM_TPU_ADMIT_OFF_AB_MS
# A message-queue wait starts admitting once it has lasted this long (the long wait for a step's tokens).
OPEN_AFTER_NS = 2_000_000
LOG_EVERY = 200

_PATCH_FLAG = "_tpu_admit_off_patch"
_WAIT_START = "_tpu_admit_off_wait_start_ns"
_MAIN = threading.main_thread()

# sha1 prefix of the dedented source (from the module file, so wrappers do not matter) of the vLLM v0.31.0 functions
# whose behaviour admit_one / the hold mirror. A mismatch means the mirror may no longer be exact: not installed.
_FINGERPRINTS = {
    "Scheduler.schedule": "39da0b63f7e988e1",
    "Scheduler._get_local_prefix_cache_hit": "e8b009d6b1163b34",
    "Scheduler._handle_blocked_waiting_request": "bd835b4fadb12dec",
    "Scheduler.finish_requests": "bf10870dd66bba4d",
    "Scheduler._holds_kv_blocks": "75cbfdbf49b09de4",
}

# Request ids admitted in the wait since the last schedule() began -> schedule count when admitted.
PENDING = {}
# finish_requests calls held for PENDING ids: (ids, status), re-issued after the next schedule() returns.
HELD = []
_state = {"engine": None, "limit": 0, "seq": 0, "window_n": 0, "charge_due": 0, "stalled_seq": -1,
          "in_sched": False, "busy": False, "ab_window": None}
stats = {"pulled": 0, "admitted": 0, "backout": 0, "ineligible": 0, "held": 0, "reissued": 0, "charged": 0,
         "ab_off": 0, "admit_ns": 0, "unwritten_miss": 0, "miss_tok": 0, "wait_tok": 0}


def enabled() -> bool:
    return LIMIT > 0


def ab_on(now_ns: int | None = None) -> bool:
    """Whether the lever is on in the current A/B window (always, without AB)."""
    if AB_MS <= 0:
        return True
    k = (time.time_ns() if now_ns is None else now_ns) // (AB_MS * 1_000_000)
    if k != _state["ab_window"]:
        _state["ab_window"] = k
        logger.info(
            "ADMIT_OFF AB window=%d on=%d admitted=%d pulled=%d backout=%d held=%d",
            k, int(k % 2 == 0), stats["admitted"], stats["pulled"], stats["backout"], stats["held"],
        )
    return k % 2 == 0


def source_fingerprints() -> dict:
    """{qualname: sha1-16} of the Scheduler functions in _FINGERPRINTS, read from vLLM's scheduler module source."""
    from vllm.v1.core.sched import scheduler as mod

    src = inspect.getsource(sys.modules[mod.__name__])
    lines = src.split("\n")
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler":
            for fn in node.body:
                key = "Scheduler." + getattr(fn, "name", "")
                if key in _FINGERPRINTS:
                    seg = textwrap.dedent("\n".join(lines[fn.lineno - 1:fn.end_lineno]) + "\n")
                    out[key] = hashlib.sha1(seg.encode()).hexdigest()[:16]
    return out


def check_fingerprints() -> list:
    got = source_fingerprints()
    return sorted(k for k, v in _FINGERPRINTS.items() if got.get(k) != v)


def pull_add(q):
    """The input queue's head item if it is an ADD, else None (never takes an item past a non-ADD)."""
    from vllm.v1.engine import EngineCoreRequestType

    with q.mutex:
        if not q.queue or q.queue[0][0] != EngineCoreRequestType.ADD:
            return None
        item = q._get()
        q.not_full.notify()
    return item


def ineligible(sched, request) -> str | None:
    """Why the waiting scan's admission of ``request`` is not the narrow async-load case (None if it is). No side
    effects. The scheduler-level checks mirror what the next pass would see where they can be known now."""
    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.request import RequestStatus

    if request.status != RequestStatus.WAITING:
        return "status"
    if request.num_computed_tokens or request.num_preemptions:
        return "not_fresh"
    if (request.resumable or request.has_encoder_inputs or request.lora_request is not None
            or request.use_structured_output or request.pooling_params is not None):
        return "kind"
    if request.num_stale_output_tokens or request.skip_reading_prefix_cache:
        return "stale_or_skip"
    if not kv_load_pacing.starts_remote_load(request):
        return "no_remote_load"
    if request.request_id in PENDING:
        return "pending"
    if any(m.req_to_blocks.get(request.request_id) for m in sched.kv_cache_manager.coordinator.single_type_managers):
        return "holds_blocks"
    if sched.pause_state != PauseState.UNPAUSED:
        return "paused"
    if len(sched.running) + sched.num_waiting_for_streaming_input >= sched.max_num_active_reqs:
        return "no_slot"
    spec = sched.vllm_config.speculative_config
    draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
    n = len(sched.running)
    if (n * (1 + sched.num_spec_tokens) >= sched.max_num_scheduled_tokens
            or n * (1 + sched.num_spec_tokens + draft_slots) >= sched.scheduler_config.max_num_batched_tokens - draft_slots):
        return "no_budget"
    return None


def _attention_group(sched) -> tuple:
    """(group id, block size) of the first full-attention KV cache group, cached per scheduler; (None, 0) if none."""
    g = sched.__dict__.get("_tpu_admit_off_attn_group")
    if g is None:
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

        g = (None, 0)
        for i, m in enumerate(sched.kv_cache_manager.coordinator.single_type_managers):
            if isinstance(m, FullAttentionManager):
                g = (i, m.block_size)
                break
        sched.__dict__["_tpu_admit_off_attn_group"] = g
    return g


def unwritten_miss_tokens(sched, request, num_hit_tokens: int) -> int | None:
    """None unless the attention group's next block after the local hit is cached but still in the same-step patch's
    unwritten set; else the tokens a hit would have covered, walking forward while that group's hashes stay cached
    (capped at the scan's num_tokens - 1). Read-only; one map lookup unless it fires. Indicator: one group's hashes."""
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    pool = sched.kv_cache_manager.block_pool
    unwritten = pool.__dict__.get("_tpu_unwritten_block_ids")
    if not unwritten:
        return None
    gid, bsize = _attention_group(sched)
    if gid is None or bsize % pool.hash_block_size:
        return None
    scale = bsize // pool.hash_block_size
    hashes = request.block_hashes
    get = pool.cached_block_hash_to_block.get_one_block

    def cached(i):
        j = (i + 1) * scale - 1
        return get(make_block_hash_with_group_id(hashes[j], gid)) if j < len(hashes) else None

    i = num_hit_tokens // bsize
    first = cached(i)
    if first is None or first.block_id not in unwritten:
        return None
    n = 1
    while cached(i + n) is not None:
        n += 1
    return min(n * bsize, max(0, request.num_tokens - 1 - num_hit_tokens))


def _back_out(request, boundary_before, why: str) -> bool:
    request.shared_prefix_boundary = boundary_before
    stats["backout"] += 1
    logger.debug("ADMIT_OFF back-out req_id=%s (%s)", request.request_id, why)
    return False


def admit_one(sched, request) -> bool:
    """vLLM's WAITING-loop admission (``Scheduler.schedule``, the ``num_computed_tokens == 0`` branch through the
    ``load_kv_async`` skip) for one fresh request at the head of the waiting queue. Returns True once it has started the
    async remote-KV load; returns False, with nothing changed, if the load would not be async or the blocks do not fit.
    Preconditions (``ineligible`` is None): no encoder inputs, LoRA, replay window or EC connector."""
    from vllm.v1.request import RequestStatus

    kvm = sched.kv_cache_manager
    connector = sched.connector
    request_id = request.request_id
    boundary_before = request.shared_prefix_boundary

    num_external_computed_tokens = 0
    load_kv_async = False
    # Get already-cached tokens.
    (
        new_computed_blocks,
        num_new_local_computed_tokens,
        request.shared_prefix_boundary,
        hit_diverged,
    ) = sched._get_local_prefix_cache_hit(request)
    # Indicator only (read-only; before the allocation can evict anything): did the hit stop at an unwritten block?
    miss = unwritten_miss_tokens(sched, request, num_new_local_computed_tokens)
    # Get externally-cached tokens (block-aligned local hit presented to the connector).
    partial_tail = num_new_local_computed_tokens % sched.block_size
    block_aligned_local = num_new_local_computed_tokens - partial_tail
    ext_tokens, load_kv_async = connector.get_num_new_matched_tokens(request, block_aligned_local)
    if request.skip_reading_prefix_cache:
        ext_tokens, load_kv_async = 0, False
    if ext_tokens is None:
        return _back_out(request, boundary_before, "ext_none")
    if partial_tail and ext_tokens > partial_tail:
        new_computed_blocks = kvm.truncate_computed_blocks(new_computed_blocks, block_aligned_local)
        num_new_local_computed_tokens = block_aligned_local
        num_external_computed_tokens = ext_tokens
    elif partial_tail:
        num_external_computed_tokens = 0
        load_kv_async = False
    else:
        num_external_computed_tokens = ext_tokens
    # (A diverged hit is reconciled only when no external tokens back it, i.e. never on the async path.)
    if not load_kv_async or num_external_computed_tokens <= 0:
        return _back_out(request, boundary_before, "not_async")
    connector_prefix_cache_queries = request.num_tokens - num_new_local_computed_tokens
    connector_prefix_cache_hits = num_external_computed_tokens

    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
    assert num_computed_tokens <= request.num_tokens

    # KVTransfer: loading remote KV, no new tokens and no lookahead slots; reserve as the scan does.
    reserved_blocks = sched._inflight_prefill_reserved_blocks() + sched._spec_decode_step_blocks()
    new_blocks = kvm.allocate_slots(
        request,
        0,
        num_new_computed_tokens=num_new_local_computed_tokens,
        new_computed_blocks=new_computed_blocks,
        num_lookahead_tokens=0,
        num_external_computed_tokens=num_external_computed_tokens,
        delay_cache_blocks=True,
        num_encoder_tokens=0,
        full_sequence_must_fit=sched.scheduler_reserve_full_isl,
        reserved_blocks=reserved_blocks,
        has_scheduled_reqs=bool(sched.running),
    )
    if new_blocks is None:
        return _back_out(request, boundary_before, "no_blocks")

    # Committed from here on (the scan sets prefill_stats before allocating; it is a record, set here once).
    if request.prefill_stats and request.num_preemptions <= 0:
        assert num_computed_tokens <= request.num_prompt_tokens
        request.prefill_stats.set(
            num_prompt_tokens=request.num_prompt_tokens,
            num_local_cached_tokens=num_new_local_computed_tokens,
            num_external_cached_tokens=num_external_computed_tokens,
        )
    connector.update_state_after_alloc(request, kvm.get_blocks(request_id), num_external_computed_tokens)
    if sched.connector_prefix_cache_stats is not None and connector_prefix_cache_queries != 0:
        sched.connector_prefix_cache_stats.record(
            num_tokens=connector_prefix_cache_queries,
            num_hits=connector_prefix_cache_hits,
            preempted=request.num_preemptions > 0,
        )
    kvm.record_prefix_cache_stats(request, num_new_local_computed_tokens)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = num_computed_tokens
    sched._inflight_prefills.add(request)
    if sched.needs_kv_cache_zeroing:
        sched._skip_zero_block_ids.update(
            kvm.get_zeroing_block_ids_in_range(request_id, num_new_local_computed_tokens, num_computed_tokens)
        )
    # The scan's skip path: out of the waiting queue, into the KV-holding queue (behind the requests already there,
    # where the next pass's scan of the KV-holding queue would leave it).
    popped = sched.waiting.pop_request()
    assert popped is request
    sched.deferred_waiting.add(request)
    sched.kv_holding_waiting.add_request(request)
    stats["wait_tok"] += request.num_prompt_tokens
    if miss is not None:
        stats["unwritten_miss"] += 1
        stats["miss_tok"] += miss
    return True


def step_window() -> None:
    """One unit of admission work in the main thread's long wait (see the module docstring)."""
    engine = _state["engine"]
    if engine is None or _state["busy"] or _state["in_sched"]:
        return
    if _state["window_n"] >= _state["limit"] or _state["stalled_seq"] == _state["seq"]:
        return
    if not ab_on():
        stats["ab_off"] += 1
        return
    sched = engine.scheduler
    _state["busy"] = True
    try:
        if not sched.waiting:
            item = pull_add(engine.input_queue)
            if item is None:
                return
            stats["pulled"] += 1
            engine._handle_client_request(*item)
            if not sched.waiting:
                return
        request = sched.waiting.peek_request()
        why = ineligible(sched, request)
        if why is not None:
            stats["ineligible"] += 1
            _state["stalled_seq"] = _state["seq"]
            return
        t0 = time.perf_counter_ns()
        if not admit_one(sched, request):
            _state["stalled_seq"] = _state["seq"]
            return
        stats["admit_ns"] += time.perf_counter_ns() - t0
        PENDING[request.request_id] = _state["seq"]
        _state["window_n"] += 1
        stats["admitted"] += 1
        if stats["admitted"] == 1 or stats["admitted"] % LOG_EVERY == 0:
            # The runtime line (Car H's gating row): logged by the thread that admits, at the first admission done in the
            # wait and every LOG_EVERY after.
            logger.info(
                "ADMIT_AHEAD ahead=%d pulled=%d backout=%d ineligible=%d held=%d reissued=%d charged=%d admit_ms_mean=%.3f "
                "unwritten_miss=%d miss_tok=%d wait_tok=%d",
                stats["admitted"], stats["pulled"], stats["backout"], stats["ineligible"], stats["held"],
                stats["reissued"], stats["charged"], stats["admit_ns"] / stats["admitted"] / 1e6,
                stats["unwritten_miss"], stats["miss_tok"], stats["wait_tok"],
            )
    finally:
        _state["busy"] = False


def note_wait(cond) -> None:
    """At each SpinCondition.wait(): once the main thread's wait has lasted OPEN_AFTER_NS, one unit of work."""
    if threading.current_thread() is not _MAIN:
        return
    now = time.monotonic_ns()
    start = getattr(cond, _WAIT_START, 0)
    if not start:
        setattr(cond, _WAIT_START, now)
    elif now - start >= OPEN_AFTER_NS:
        step_window()


def note_read(cond) -> None:
    if threading.current_thread() is _MAIN:
        setattr(cond, _WAIT_START, 0)


def make_schedule(orig):
    def schedule(self, *args, **kwargs):
        _state["in_sched"] = True
        _state["seq"] += 1
        _state["charge_due"] = _state["window_n"]
        _state["window_n"] = 0
        try:
            out = orig(self, *args, **kwargs)
        finally:
            _state["in_sched"] = False
            _state["charge_due"] = 0
            PENDING.clear()
        if HELD:
            held = list(HELD)
            HELD.clear()
            for ids, status in held:
                stats["reissued"] += len(ids)
                self.finish_requests(list(ids), status)
        return out

    setattr(schedule, _PATCH_FLAG, True)
    schedule.__wrapped__ = orig
    return schedule


def make_handle_blocked(orig):
    def _handle_blocked_waiting_request(self, request) -> bool:
        due = _state["charge_due"]
        if due:
            _state["charge_due"] = 0
            state = self.__dict__.get(kv_load_pacing._STATE)
            if state and state[0]:
                state[1] += due
                stats["charged"] += due
        return orig(self, request)

    setattr(_handle_blocked_waiting_request, _PATCH_FLAG, True)
    _handle_blocked_waiting_request.__wrapped__ = orig
    return _handle_blocked_waiting_request


def make_finish_requests(orig):
    def finish_requests(self, request_ids, finished_status):
        if PENDING and request_ids is not None:
            ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
            held = tuple(i for i in ids if i in PENDING)
            if held:
                HELD.append((held, finished_status))
                stats["held"] += len(held)
                rest = [i for i in ids if i not in PENDING]
                return orig(self, rest, finished_status) if rest else []
        return orig(self, request_ids, finished_status)

    setattr(finish_requests, _PATCH_FLAG, True)
    finish_requests.__wrapped__ = orig
    return finish_requests


def engage_reason(engine) -> str | None:
    """Why this engine core does not take the lever (None if it does)."""
    import queue

    from vllm.v1.core.sched.request_queue import SchedulingPolicy
    from vllm.v1.core.sched.scheduler import Scheduler

    sched = getattr(engine, "scheduler", None)
    if not isinstance(sched, Scheduler):
        return "scheduler is not vLLM's Scheduler"
    for name in ("schedule", "finish_requests", "_handle_blocked_waiting_request", "_get_local_prefix_cache_hit"):
        if getattr(type(sched), name) is not getattr(Scheduler, name):
            return "%s overrides Scheduler.%s" % (type(sched).__name__, name)
    transfer = sched.vllm_config.kv_transfer_config
    if sched.connector is None or transfer is None or transfer.kv_role != "kv_consumer":
        return "not a KV consumer"
    if sched.vllm_config.parallel_config.data_parallel_size != 1:
        return "data parallel engine"
    if sched.policy != SchedulingPolicy.FCFS:
        return "scheduling policy is not FCFS"
    if sched.prefix_replay_tokens or sched.is_encoder_decoder or sched.ec_connector is not None:
        return "replay window, encoder-decoder or EC connector"
    if not isinstance(getattr(engine, "input_queue", None), queue.Queue):
        return "no input queue"
    return None


def attach(engine) -> None:
    """Registers the engine core (EngineCoreProc.__init__ wrapper) and logs the engagement line once."""
    why = engage_reason(engine)
    if why is not None:
        logger.warning("ADMIT_OFF not engaged: %s; admissions run in schedule() as before", why)
        return
    sched = engine.scheduler
    pcap = pacing_cap(sched)
    _state["limit"] = min(LIMIT, pcap) if pcap else LIMIT
    _state["engine"] = engine
    # The engagement line: logged by the engine core process once its engine is built (patch-time lines are logged
    # before its logging reaches the served log).
    logger.info(
        "ADMIT_OFF engaged: remote-KV admissions run in the engine core's token wait "
        "(up to %d between schedules, after %.1f ms of a wait; KV load pacing cap %d; AB %d ms)",
        _state["limit"], OPEN_AFTER_NS / 1e6, pcap, AB_MS,
    )


def pacing_cap(sched) -> int:
    """The KV load pacing cap this scheduler's passes enforce (0 if the pacing patch is not installed)."""
    fn, seen = type(sched)._handle_blocked_waiting_request, []
    while fn is not None and fn not in seen:
        if getattr(fn, kv_load_pacing._PATCH_FLAG, False):
            return kv_load_pacing._scheduler_cap(sched, envs.VLLM_TPU_KV_LOAD_STARTS_PER_STEP)
        seen.append(fn)
        fn = getattr(fn, "__wrapped__", None)
    return 0


def patch_admit_off() -> None:
    """engine_core stage; a no-op unless VLLM_TPU_ADMIT_OFF > 0."""
    if not enabled():
        return
    from vllm.distributed.device_communicators import shm_broadcast
    from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCoreProc

    if getattr(Scheduler.finish_requests, _PATCH_FLAG, False):
        return
    if getattr(shm_broadcast, "SPINLOOP_EXT_ENABLED", False):
        # Reads spin in C before the first wait(); the wait hook would never run.
        logger.warning("VLLM_TPU_ADMIT_OFF refused: VLLM_USE_SPINLOOP_EXT is on; admissions run in schedule() as before")
        return
    mismatched = check_fingerprints()
    if mismatched:
        logger.warning("VLLM_TPU_ADMIT_OFF refused: vLLM source changed for %s; admissions run in schedule() as before",
                       ", ".join(mismatched))
        return
    Scheduler.schedule = make_schedule(Scheduler.schedule)
    Scheduler._handle_blocked_waiting_request = make_handle_blocked(Scheduler._handle_blocked_waiting_request)
    Scheduler.finish_requests = make_finish_requests(Scheduler.finish_requests)

    orig_wait = SpinCondition.wait
    orig_record_read = SpinCondition.record_read

    def wait(self, timeout_ms=None):
        note_wait(self)
        return orig_wait(self, timeout_ms)

    def record_read(self):
        note_read(self)
        return orig_record_read(self)

    for fn, orig in ((wait, orig_wait), (record_read, orig_record_read)):
        setattr(fn, _PATCH_FLAG, True)
        fn.__wrapped__ = orig
    SpinCondition.wait = wait
    SpinCondition.record_read = record_read

    init = EngineCoreProc.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        attach(self)

    setattr(__init__, _PATCH_FLAG, True)
    __init__.__wrapped__ = init
    EngineCoreProc.__init__ = __init__
    logger.info(
        "VLLM_TPU_ADMIT_OFF=%d: remote-KV admissions run in the engine core's token wait (after %.1f ms; AB %d ms)",
        LIMIT, OPEN_AFTER_NS / 1e6, AB_MS,
    )
