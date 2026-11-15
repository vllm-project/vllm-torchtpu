# SPDX-License-Identifier: Apache-2.0
"""Host timeline of a PD decode's admission path (VLLM_TPU_HOSTADMIT_PROBE).

The variable names profile windows, ``"<start_s>:<duration_s>[@<hz>][i][+...]"``,
in seconds after the first Stage-3 load a process sees (on a KV producer: the
first Stage-3 send registration in a worker, the first new request in the
engine core). From that load until
the last window ends, the engine core and every worker record the host time
of the functions on the admission path (scheduler, KV cache manager, KV
connector, Stage-3 admission, staging repack, message queues), the runner's
host_phases marks, every garbage collection, and one row of counts per step.
In each window py-spy (VLLM_TPU_HOSTADMIT_PYSPY, the binary) samples the
engine core (``py-spy record`` at ``hz``, default 400, with idle threads when
the window ends in ``i``) and the worker ranks in
VLLM_TPU_HOSTADMIT_PYSPY_RANKS ("0+3" or "all") with repeated ``py-spy dump``
at up to ``hz``. Files land in ``$CDK_OUTPUT_DIR/hostadmit``; the event
buffers are written once after each run of adjacent windows.

Unset (the default), nothing is wrapped and every hook returns at once.
Functions called once per running request per step (``hot``) are summed per
step into one ``<name>#sum`` event (argument: the call count) ending at the
next step's start; calls above ``_HOT_NS`` or with an argument of at least
``_HOT_ARG`` are kept one by one, so the buffers hold a full served run.
"""

import contextlib
import ctypes
import gc
import itertools
import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_RATE_HZ = 400
_HOT_NS = 50_000
_HOT_ARG = 64
_EVENTS = 4_000_000
_STEPS = 400_000

# Per-step row: perf_counter_ns, process kind (0 engine core, 1 worker), new
# requests, scheduled tokens, Stage-3 loads named, repacks named, prompt
# tokens of the new requests, block ids of the new requests, cached
# requests, finished requests, then four process-specific counters
# (_ENGINE_EXTRA or _WORKER_EXTRA).
_STEP_COLS = (
    "t_ns",
    "kind",
    "new_reqs",
    "tokens",
    "loads",
    "repacks",
    "new_prompt_tokens",
    "new_block_ids",
    "cached_reqs",
    "finished",
    "x0",
    "x1",
    "x2",
    "x3",
)
_ENGINE_EXTRA = ("running", "waiting", "free_blocks", "unused")
_WORKER_EXTRA = (
    "staging_free",
    "staging_quarantined",
    "pending_repacks",
    "parked_loads",
)
_PRODUCER_EXTRA = (
    "registered_sends",
    "send_outcomes",
    "terminal_cleanup",
    "done_sending",
)


def _windows(spec: str) -> list[tuple[float, float, int, bool]]:
    """(start_s, duration_s, rate_hz, idle) per window, in start order."""
    out = []
    for part in spec.replace(",", "+").split("+"):
        part = part.strip()
        if not part:
            continue
        idle = part.endswith("i")
        part = part.rstrip("i")
        part, _, rate = part.partition("@")
        start, _, dur = part.partition(":")
        out.append((float(start), float(dur or 60), int(rate or _RATE_HZ), idle))
    return sorted(out)


def _dump_times(windows) -> list[float]:
    """One dump 15 s after each run of windows less than 30 s apart."""
    times = []
    for start, dur, _, _ in windows:
        if times and start <= times[-1] - 15 + 30:
            times[-1] = max(times[-1], start + dur + 15)
        else:
            times.append(start + dur + 15)
    return times


def _pyspy_ranks(spec: str) -> set[int] | None:
    """Worker ranks py-spy samples; None means every rank."""
    if spec.strip() == "all":
        return None
    return {int(r) for r in spec.replace(",", "+").split("+") if r.strip()}


_WINDOWS = _windows(envs.VLLM_TPU_HOSTADMIT_PROBE)
ENABLED = bool(_WINDOWS)


class _State:
    def __init__(self) -> None:
        self.role = ""
        self.rank = -1
        self.attached = False
        self.recording = False
        self.done = False
        self.anchor_ns = 0
        self.anchor_wall_ns = 0
        self.end_ns = 0
        self.dumps: list[int] = []
        self.names: list[str] = []
        self.ids: dict[str, int] = {}
        self.counter = itertools.count()
        self.step_counter = itertools.count()
        self.dropped = 0
        self.t0 = np.zeros(0, np.int64)
        self.dur = np.zeros(0, np.int64)
        self.nid = np.zeros(0, np.int32)
        self.arg = np.zeros(0, np.int64)
        self.main = np.zeros(0, np.int8)
        self.steps = np.zeros((0, len(_STEP_COLS)), np.int64)
        self.hot_ns: dict[int, int] = {}
        self.hot_n: dict[int, int] = {}
        self.main_ident = threading.main_thread().ident
        self.last_event = 0
        self.last_step = 0
        self.part = 0
        self.gc_start = 0
        self.extra = None
        self.producer = False
        self.connector_attached = False
        self.extra_cols: tuple[str, ...] = ()


_S = _State()


def _outdir() -> str:
    base = os.environ.get("CDK_OUTPUT_DIR") or "/tmp"
    return os.path.join(base, "hostadmit")


def _nid(name: str) -> int:
    i = _S.ids.get(name)
    if i is None:
        i = len(_S.names)
        _S.names.append(name)
        _S.ids[name] = i
    return i


def _record(nid: int, t0: int, t1: int, arg: int = 0) -> None:
    i = next(_S.counter)
    if i >= _EVENTS:
        _S.dropped += 1
        return
    _S.t0[i] = t0
    _S.dur[i] = t1 - t0
    _S.nid[i] = nid
    _S.arg[i] = arg
    _S.main[i] = threading.get_ident() == _S.main_ident


def _wrap(
    obj,
    attr: str,
    label: str | None = None,
    argfn=None,
    hot: bool = False,
    anchor: bool = False,
):
    fn = obj.__dict__.get(attr) if isinstance(obj, type) else getattr(obj, attr, None)
    if fn is None or not callable(fn) or getattr(fn, "_hostadmit", False):
        return
    owner = obj.__name__ if isinstance(obj, type) else type(obj).__name__
    nid = _nid(label or f"{owner}.{attr}")
    agg = _nid(f"{label or f'{owner}.{attr}'}#sum") if hot else -1
    perf = time.perf_counter_ns

    def wrapper(*args, **kwargs):
        if not _S.recording:
            if not anchor or _S.done:
                return fn(*args, **kwargs)
            _start_recording(perf())
        t0 = perf()
        try:
            return fn(*args, **kwargs)
        finally:
            t1 = perf()
            arg = 0
            if argfn is not None:
                try:
                    arg = int(argfn(args, kwargs))
                except Exception:
                    arg = -1
            if hot and t1 - t0 < _HOT_NS and arg < _HOT_ARG:
                _S.hot_ns[agg] = _S.hot_ns.get(agg, 0) + t1 - t0
                _S.hot_n[agg] = _S.hot_n.get(agg, 0) + 1
            else:
                _record(nid, t0, t1, arg)

    wrapper._hostadmit = True
    wrapper.__wrapped__ = fn
    setattr(obj, attr, wrapper)


def _wrap_all(obj, attrs, prefix: str) -> None:
    if obj is None:
        return
    for attr in attrs:
        _wrap(obj, attr, f"{prefix}.{attr}")


def _arg(index: int, name: str):
    def get(args, kwargs):
        if name in kwargs:
            return kwargs[name]
        return args[index] if len(args) > index else 0

    return get


def _len_arg(index: int, name: str):
    def get(args, kwargs):
        value = (
            kwargs[name]
            if name in kwargs
            else (args[index] if len(args) > index else ())
        )
        return len(value or ())

    return get


def _gc_callback(phase: str, info: dict) -> None:
    if not _S.recording:
        return
    now = time.perf_counter_ns()
    if phase == "start":
        _S.gc_start = now
    elif _S.gc_start:
        _record(
            _nid(f"gc.gen{info.get('generation', -1)}"),
            _S.gc_start,
            now,
            info.get("collected", 0),
        )
        _S.gc_start = 0


def _common_attach(role: str, rank: int) -> None:
    global _EVENTS
    _EVENTS = envs.VLLM_TPU_HOSTADMIT_EVENTS or _EVENTS
    _S.role = role
    _S.rank = rank
    _S.t0 = np.zeros(_EVENTS, np.int64)
    _S.dur = np.zeros(_EVENTS, np.int64)
    _S.nid = np.zeros(_EVENTS, np.int32)
    _S.arg = np.zeros(_EVENTS, np.int64)
    _S.main = np.zeros(_EVENTS, np.int8)
    _S.steps = np.zeros((_STEPS, len(_STEP_COLS)), np.int64)
    gc.callbacks.append(_gc_callback)
    _wrap_message_queue()
    _S.attached = True
    logger.info(
        "HOSTADMIT probe attached role=%s rank=%d windows=%s", role, rank, _WINDOWS
    )


def _wrap_message_queue() -> None:
    try:
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    except Exception:
        return
    _wrap(MessageQueue, "enqueue", "mq.enqueue")
    _wrap(MessageQueue, "dequeue", "mq.dequeue")
    orig = MessageQueue.__dict__.get("acquire_read")
    if orig is None or getattr(orig, "_hostadmit", False):
        return
    nid = _nid("mq.acquire_read_wait")

    @contextlib.contextmanager
    def acquire_read(self, *args, **kwargs):
        t0 = time.perf_counter_ns()
        with orig(self, *args, **kwargs) as buf:
            if _S.recording:
                _record(nid, t0, time.perf_counter_ns())
            yield buf

    acquire_read._hostadmit = True
    MessageQueue.acquire_read = acquire_read


def _meta_counts(scheduler_output) -> tuple[int, int]:
    meta = getattr(scheduler_output, "kv_connector_metadata", None)
    if meta is None:
        return 0, 0
    loads = getattr(meta, "reqs_to_load", None) or ()
    repacks = getattr(meta, "qwen_repack_req_ids", None) or ()
    return len(loads), len(repacks)


def _step(scheduler_output, kind: int) -> None:
    """Counts one step and moves the probe through its windows."""
    now = time.perf_counter_ns()
    loads, repacks = _meta_counts(scheduler_output)
    if not _S.recording:
        if _S.done:
            return
        first_request = (
            _S.producer and kind == 0 and scheduler_output.scheduled_new_reqs
        )
        if loads == 0 and not first_request:
            return
        _start_recording(now)
    if _S.hot_ns:
        for nid, total in _S.hot_ns.items():
            _record(nid, now - total, now, _S.hot_n.get(nid, 0))
        _S.hot_ns.clear()
        _S.hot_n.clear()
    new = scheduler_output.scheduled_new_reqs
    prompt = 0
    blocks = 0
    for req in new:
        if req.prompt_token_ids is not None:
            prompt += len(req.prompt_token_ids)
        for ids in req.block_ids or ():
            blocks += len(ids)
    cached = scheduler_output.scheduled_cached_reqs
    extra = (0, 0, 0, 0)
    if _S.extra is not None:
        try:
            extra = _S.extra()
        except Exception:
            extra = (-1, -1, -1, -1)
    i = next(_S.step_counter)
    if i < _STEPS:
        _S.steps[i] = (
            now,
            kind,
            len(new),
            scheduler_output.total_num_scheduled_tokens,
            loads,
            repacks,
            prompt,
            blocks,
            len(getattr(cached, "req_ids", ()) or ()),
            len(scheduler_output.finished_req_ids or ()),
            *extra,
        )
    if _S.dumps and now >= _S.dumps[0]:
        _S.dumps.pop(0)
        _dump(final=not _S.dumps)


def _start_recording(now: int) -> None:
    _S.anchor_ns = now
    _S.anchor_wall_ns = time.time_ns()
    dumps = _dump_times(_WINDOWS)
    _S.dumps = [now + int(t * 1e9) for t in dumps]
    _S.recording = True
    logger.info(
        "HOSTADMIT probe role=%s rank=%d: first Stage-3 load, dumps at +%s s",
        _S.role,
        _S.rank,
        dumps,
    )
    binary = envs.VLLM_TPU_HOSTADMIT_PYSPY
    ranks = _pyspy_ranks(envs.VLLM_TPU_HOSTADMIT_PYSPY_RANKS)
    if binary and (_S.role == "engine" or ranks is None or _S.rank in ranks):
        threading.Thread(
            target=_pyspy_loop, args=(binary,), name="hostadmit-pyspy", daemon=True
        ).start()


def _allow_ptrace() -> None:
    # PR_SET_PTRACER_ANY, so a Yama ptrace_scope of 1 still lets the child
    # py-spy read this process.
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(
            ctypes.c_int(0x59616D61),
            ctypes.c_ulong(2**64 - 1),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
    except Exception as exc:
        logger.info("HOSTADMIT prctl failed: %s", exc)


def _pyspy_loop(binary: str) -> None:
    _allow_ptrace()
    tag = f"{_S.role}-pid{os.getpid()}"
    local = "/tmp/hostadmit"
    os.makedirs(local, exist_ok=True)
    if not os.access(binary, os.X_OK):
        # A mapped directory may drop the executable bit.
        copy = os.path.join(local, f"py-spy-{os.getpid()}")
        shutil.copy(binary, copy)
        os.chmod(copy, 0o755)
        binary = copy
    for index, (start, dur, rate, idle) in enumerate(_WINDOWS):
        wait = (_S.anchor_ns + int(start * 1e9) - time.perf_counter_ns()) / 1e9
        if wait > 0:
            time.sleep(wait)
        if _S.role != "engine":
            _dump_window(binary, tag, local, index, start, dur, rate)
            continue
        out = os.path.join(local, f"{tag}-w{index:02d}.speedscope.json")
        cmd = [
            binary,
            "record",
            "--pid",
            str(os.getpid()),
            "--rate",
            str(rate),
            "--duration",
            str(int(dur)),
            "--format",
            "speedscope",
            "--nonblocking",
            "--output",
            out,
        ]
        if idle:
            cmd.append("--idle")
        meta = {
            "cmd": cmd,
            "window": [start, dur, rate, idle],
            "start_wall_ns": time.time_ns(),
            "start_perf_ns": time.perf_counter_ns(),
            "anchor_wall_ns": _S.anchor_wall_ns,
            "anchor_perf_ns": _S.anchor_ns,
        }
        logger.info(
            "HOSTADMIT py-spy start %s window %d: %s", tag, index, " ".join(cmd)
        )
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=dur + 300, check=False
            )
            meta["returncode"] = proc.returncode
            meta["stdout"] = proc.stdout[-4000:]
            meta["stderr"] = proc.stderr[-4000:]
        except Exception as exc:
            meta["error"] = repr(exc)
        meta["end_wall_ns"] = time.time_ns()
        meta["end_perf_ns"] = time.perf_counter_ns()
        meta_path = os.path.join(local, f"{tag}-w{index:02d}.meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        logger.info(
            "HOSTADMIT py-spy done %s window %d rc=%s err=%s",
            tag,
            index,
            meta.get("returncode"),
            meta.get("error") or meta.get("stderr", "")[-300:],
        )
        # Copied off this thread so the next window starts on time.
        threading.Thread(
            target=_copy_out, args=([meta_path, out],), daemon=True
        ).start()


def _dump_window(binary, tag, local, index, start, dur, rate) -> None:
    """Samples this process with repeated ``py-spy dump`` for one window.

    ``py-spy record`` spends about 40 s attaching to a worker (1,700+ OS
    threads) before its first sample, while one dump takes about 0.2 s, so
    workers sample with dumps: every thread's stack, each block headed by
    ``=== <time_ns> <perf_counter_ns>``, at up to ``rate`` Hz.
    """
    out = os.path.join(local, f"{tag}-w{index:02d}.dumps.txt")
    end = _S.anchor_ns + int((start + dur) * 1e9)
    period = 1.0 / max(1, rate)
    cmd = [binary, "dump", "--pid", str(os.getpid()), "--nonblocking"]
    n = 0
    with open(out, "w") as f:
        while time.perf_counter_ns() < end:
            t = time.perf_counter()
            f.write(f"=== {time.time_ns()} {time.perf_counter_ns()}\n")
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=10, check=False
                )
                f.write(proc.stdout)
            except Exception as exc:
                f.write(f"error {exc!r}\n")
            n += 1
            time.sleep(max(0.0, period - (time.perf_counter() - t)))
    logger.info("HOSTADMIT py-spy dumps %s window %d: %d samples", tag, index, n)
    threading.Thread(target=_copy_out, args=([out],), daemon=True).start()


def _copy_out(paths: list[str]) -> None:
    dst = _outdir()
    try:
        os.makedirs(dst, exist_ok=True)
        for path in paths:
            if os.path.exists(path):
                shutil.copy(path, os.path.join(dst, os.path.basename(path)))
    except Exception as exc:
        logger.warning("HOSTADMIT copy to %s failed: %s", dst, exc)


def _dump(final: bool) -> None:
    n_events = min(next(_S.counter), _EVENTS)
    # The counter moved past n_events once more; the extra index is unused.
    _S.counter = itertools.count(n_events)
    n_steps = min(next(_S.step_counter), _STEPS)
    _S.step_counter = itertools.count(n_steps)
    lo, slo = _S.last_event, _S.last_step
    _S.last_event, _S.last_step = n_events, n_steps
    if final:
        _S.recording = False
        _S.done = True
    tag = f"{_S.role}-pid{os.getpid()}-part{_S.part}"
    _S.part += 1
    local = os.path.join("/tmp/hostadmit", f"{tag}.npz")
    os.makedirs("/tmp/hostadmit", exist_ok=True)
    np.savez(
        local,
        t0=_S.t0[lo:n_events],
        dur=_S.dur[lo:n_events],
        nid=_S.nid[lo:n_events],
        arg=_S.arg[lo:n_events],
        main=_S.main[lo:n_events],
        steps=_S.steps[slo:n_steps],
        step_cols=np.array(_STEP_COLS),
        extra_cols=np.array(_S.extra_cols),
        names=np.array(_S.names),
        clock=np.array(
            [_S.anchor_wall_ns, _S.anchor_ns, time.time_ns(), time.perf_counter_ns()],
            np.int64,
        ),
        info=np.array([_S.role, str(_S.rank), str(_S.dropped)]),
    )
    logger.info(
        "HOSTADMIT dump %s: events %d..%d steps %d..%d dropped %d",
        tag,
        lo,
        n_events,
        slo,
        n_steps,
        _S.dropped,
    )
    threading.Thread(target=_copy_out, args=([local],), daemon=True).start()


# --- engine core -----------------------------------------------------------

_SCHEDULER_ATTRS = (
    "update_from_output",
    "add_request",
    "finish_requests",
    "_update_waiting_for_remote_kv",
    "_update_from_kv_xfer_finished",
    "_free_request",
    "_make_cached_request_data",
    "_update_after_schedule",
    "_build_kv_connector_meta",
    "_handle_blocked_waiting_request",
    "_preempt_request",
    "_connector_finished",
    "get_grammar_bitmask",
    "make_stats",
)
_KV_MANAGER_ATTRS = (
    "get_computed_blocks",
    "get_computed_blocks_for_connector",
    "free",
    "cache_blocks",
    "take_events",
    "remove_skipped_blocks",
    "evict_blocks",
    "new_step_starts",
)
_BLOCK_POOL_ATTRS = ("cache_full_blocks", "free_blocks", "evict_blocks")
_CONNECTOR_ATTRS = (
    "get_num_new_matched_tokens",
    "update_state_after_alloc",
    "build_connector_meta",
    "update_connector_output",
    "request_finished",
    "request_finished_all_groups",
    "take_events",
)
_CONNECTOR_SCHEDULER_ATTRS = (
    "get_num_new_matched_tokens",
    "update_state_after_alloc",
    "build_connector_meta",
    "request_finished",
    "update_connector_output",
    "_stage3_register_ahead",
)


def attach_engine_core(engine) -> None:
    """Wraps the engine core's admission path (called after its __init__)."""
    if not ENABLED or _S.attached:
        return
    _common_attach("engine", -1)
    transfer = engine.vllm_config.kv_transfer_config
    _S.producer = transfer is not None and transfer.is_kv_producer
    scheduler = engine.scheduler
    schedule = scheduler.schedule
    nid = _nid("scheduler.schedule")

    def wrapped_schedule(*args, **kwargs):
        t0 = time.perf_counter_ns()
        out = schedule(*args, **kwargs)
        t1 = time.perf_counter_ns()
        if _S.recording:
            _record(nid, t0, t1, len(out.scheduled_new_reqs))
        _step(out, 0)
        return out

    wrapped_schedule._hostadmit = True
    scheduler.schedule = wrapped_schedule
    _wrap_all(scheduler, _SCHEDULER_ATTRS, "scheduler")
    manager = getattr(scheduler, "kv_cache_manager", None)
    if manager is not None:
        _wrap(
            manager,
            "allocate_slots",
            "kvm.allocate_slots",
            _arg(1, "num_new_tokens"),
            hot=True,
        )
        _wrap_all(manager, _KV_MANAGER_ATTRS, "kvm")
        pool = getattr(manager, "block_pool", None)
        if pool is not None:
            _wrap(
                pool,
                "get_new_blocks",
                "pool.get_new_blocks",
                _arg(0, "num_blocks"),
                hot=True,
            )
            _wrap_all(pool, _BLOCK_POOL_ATTRS, "pool")
    connector = getattr(scheduler, "connector", None)
    if connector is not None:
        _wrap_all(connector, _CONNECTOR_ATTRS, "conn")
        _wrap_all(
            getattr(connector, "connector_scheduler", None),
            _CONNECTOR_SCHEDULER_ATTRS,
            "connsched",
        )
    _wrap(engine, "preprocess_add_request", "engine.preprocess_add_request")
    _wrap_all(
        engine,
        ("add_request", "post_step", "_process_aborts_queue", "_process_input_queue"),
        "engine",
    )
    _wrap(engine, "step_fn", "engine.step_fn")
    executor = getattr(engine, "model_executor", None)
    _wrap_all(executor, ("execute_model", "sample_tokens"), "executor")
    pool = getattr(manager, "block_pool", None)

    def extra():
        return (
            len(scheduler.running),
            len(scheduler.waiting),
            pool.get_num_free_blocks() if pool is not None else -1,
            0,
        )

    _S.extra = extra
    _S.extra_cols = _ENGINE_EXTRA


def patch_engine_core() -> None:
    """Attaches the probe to the engine core once it is built (engine_core stage)."""
    if not ENABLED:
        return
    from vllm.v1.engine.core import EngineCoreProc

    init = EngineCoreProc.__init__
    if getattr(init, "_hostadmit", False):
        return

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        try:
            attach_engine_core(self)
        except Exception as exc:
            logger.warning("HOSTADMIT engine core attach failed: %s", exc)

    __init__._hostadmit = True
    EngineCoreProc.__init__ = __init__


# --- worker ----------------------------------------------------------------

_RUNNER_ATTRS = (
    "execute_model",
    "sample_tokens",
    "_update_states",
    "maybe_setup_kv_connector",
    "get_finished_kv_transfers",
    "_prepare_inputs",
    "_flush_disjoint_async_results",
    "kv_connector_no_forward",
)
_WORKER_CONNECTOR_ATTRS = (
    "start_load_kv",
    "get_transfer_results",
    "get_finished",
    "bind_connector_metadata",
    "handle_preemptions",
    "build_connector_worker_meta",
    "get_kv_connector_stats",
    "clear_connector_metadata",
    "wait_for_save",
    "get_block_ids_with_load_errors",
)
_CONNECTOR_WORKER_ATTRS = (
    "process_send_load",
    "_submit_stage3_loads",
    "get_finished",
    "repack_landed",
    "_stage3_batched_admission",
    "_stage3_reserve_destination",
    "_stage3_reserve_destination_reduced",
    "_dispatch_stage3_load_submit",
    "_drain_stage3_submit_outcomes",
    "_submit_stage3_load_as_leader",
    "_stage_stage3_submit_task",
    "_stage3_finish_destination",
    "_stage3_admission_after_pass",
    "_stage3_admission_reference",
    "_bind_stage3_request_ids",
)
_STAGING_ATTRS = (
    "reserve",
    "release",
    "advance_quarantine",
    "pages_needed",
    "plan_repack",
    "run_repack",
    "repack",
    "indices",
    "_run_chunks",
    "shared_passes",
)


# A KV producer starts recording at its first Stage-3 send registration.
_PRODUCER_REGISTRATION_ATTRS = (
    "_register_stage3_request_blocks_grouped",
    "_register_stage3_request_row_spans",
)
_PRODUCER_ATTRS = (
    "_stage3_publish_send",
    "_stage3_ready_producer_send_terminals",
    "_stage3_release_producer_registration",
)


def runner_step(runner, scheduler_output) -> None:
    """Counts a worker step; attaches the probe on the first one."""
    if not _S.attached:
        _attach_runner(runner)
    if not _S.connector_attached:
        _attach_connector()
    _step(scheduler_output, 1)


def _attach_runner(runner) -> None:
    from vllm_torchtpu.runner import host_phases

    _common_attach(f"rank{host_phases._rank}", host_phases._rank)
    _wrap_all(runner, _RUNNER_ATTRS, "runner")
    host_phases.set_hook(_phase_hook)


def _attach_connector() -> None:
    """Wraps the KV connector once it is installed (the warmup steps that
    attach the probe can run without one)."""
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    if not has_kv_transfer_group():
        return
    _S.connector_attached = True
    connector = get_kv_transfer_group()
    _wrap_all(connector, _WORKER_CONNECTOR_ATTRS, "wconn")
    subs = list(getattr(connector, "_connectors", None) or ())
    for i, sub in enumerate(subs):
        _wrap_all(sub, _WORKER_CONNECTOR_ATTRS, f"wconn{i}")
    worker = None
    for candidate in [connector, *subs]:
        cw = getattr(candidate, "connector_worker", None)
        if cw is not None and hasattr(cw, "_submit_stage3_loads"):
            worker = cw
            break
    _wrap_all(worker, _CONNECTOR_WORKER_ATTRS, "cworker")
    if worker is not None and getattr(worker, "is_producer", False):
        _S.producer = True
        for attr in _PRODUCER_REGISTRATION_ATTRS:
            _wrap(worker, attr, f"cworker.{attr}", anchor=True)
        _wrap_all(worker, _PRODUCER_ATTRS, "cworker")

        def producer_extra():
            return (
                len(getattr(worker, "_stage3_registered_sends", ())),
                len(getattr(worker, "_stage3_send_outcomes", ())),
                len(getattr(worker, "_stage3_terminal_cleanup", ())),
                len(getattr(worker, "_done_sending", ())),
            )

        _S.extra = producer_extra
        _S.extra_cols = _PRODUCER_EXTRA
    staging = getattr(worker, "_staging", None)
    _wrap_all(staging, _STAGING_ATTRS, "staging")
    if staging is not None:

        def extra():
            return (
                int(staging.free.sum()),
                staging.quarantined_pages(),
                len(getattr(worker, "_pending_repacks", ())),
                len(getattr(worker, "_stage3_destination_waits", ())),
            )

        _S.extra = extra
        _S.extra_cols = _WORKER_EXTRA
    try:
        from vllm.distributed.parallel_state import get_tp_group

        _wrap(get_tp_group(), "broadcast_object", "tp.broadcast_object")
    except Exception as exc:
        logger.info("HOSTADMIT tp group not wrapped: %s", exc)


_PHASE_IDS: dict[str, int] = {}


def _phase_hook(name: str, t0: int, t1: int) -> None:
    if _S.recording:
        nid = _PHASE_IDS.get(name)
        if nid is None:
            nid = _PHASE_IDS[name] = _nid(f"phase.{name}")
        _record(nid, t0, t1)
