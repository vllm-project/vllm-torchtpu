# SPDX-License-Identifier: Apache-2.0
"""A decode's remote-KV admissions run in the engine core's token wait.

See `vllm_torchtpu.core.admit_off` (q38 SV-ADMIT ADMIT-OFF / ADMIT-AHEAD). A real vLLM ``Scheduler`` runs a KV-consumer
engine whose stand-in connector loads, asynchronously, every request that carries ``kv_transfer_params``, records
every registration / metadata build / finish, and (like the Raiden consumer) leaves a registered load in
``reqs_to_load`` when a request finishes before the next metadata build. The engine core is a real
``EngineCoreProc`` without ``__init__`` (its own ``_handle_client_request`` / ``add_request``) over a real input queue.

* Differential: the same arrivals, landings and finishes driven through a stock scheduler and through one whose
  admissions run in the wait give identical connector records (block ids included), scheduler outputs, queues,
  statuses, free and cached block counts after every pass, with and without the KV load pacing cap.
* Safety: back-outs leave nothing behind; the input queue is only taken from its head and only for ADDs; finishes of
  requests admitted in the wait are held until after the next ``schedule()``; in-wait starts are charged to the next
  pass's pacing cap; ineligible requests are left to ``schedule()``; no request is admitted twice.
* Window: only the main thread's waits that have lasted ``OPEN_AFTER_NS`` work; A/B windows follow the wall clock.
* Install: key off patches nothing; key on logs one engagement line from the engine core; the Car H patch stack keeps
  the KV decode fast path installed with the key on and off.
"""

import os
import queue
import subprocess
import sys
import threading

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.device_communicators import shm_broadcast
from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu import envs
from vllm_torchtpu.core import admit_off as ao
from vllm_torchtpu.core import kv_load_pacing

BLOCK_SIZE = 16
NUM_BLOCKS = 512
PROMPT_LEN = 40
KEYS = ("VLLM_TPU_ADMIT_OFF", "VLLM_TPU_ADMIT_OFF_AB_MS")
ADD = EngineCoreRequestType.ADD
ABORT = EngineCoreRequestType.ABORT


class AdmitTestConnector(KVConnectorBase_V1, SupportsHMA):
    """Scheduler half of a P/D consumer: loads ``params["num_tokens"]`` minus the local hit, async; records events."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.log: list[tuple] = []
        self.reqs_to_load: dict[str, tuple] = {}

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params
        if not params or params.get("_remote_kv_processed"):
            return 0, False
        if params.get("unknown"):
            return None, False
        count = max(params["num_tokens"] - num_computed_tokens, 0)
        return count, count > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            ids = tuple(blocks.get_block_ids()[0])
            self.reqs_to_load[request.request_id] = ids
            self.log.append(("alloc", request.request_id, num_external_tokens, ids))
            request.kv_transfer_params["_remote_kv_processed"] = True

    def build_connector_meta(self, scheduler_output):
        self.log.append(("build", tuple(sorted(self.reqs_to_load))))
        self.reqs_to_load = {}
        return KVConnectorMetadata()

    def request_finished_all_groups(self, request, block_ids):
        # Like the Raiden consumer: a load registered but not yet built stays in reqs_to_load (a leak).
        self.log.append(("finish", request.request_id, request.request_id in self.reqs_to_load))
        return False, None

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


@pytest.fixture
def fresh(monkeypatch):
    """Restores every class attribute the patches touch and resets the module state."""
    for owner, names in ((Scheduler, ("schedule", "_handle_blocked_waiting_request", "finish_requests")),
                         (SpinCondition, ("wait", "record_read")), (EngineCoreProc, ("__init__",))):
        for name in names:
            monkeypatch.setattr(owner, name, owner.__dict__[name])
    for k in KEYS + ("VLLM_TPU_KV_LOAD_STARTS_PER_STEP",):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ao, "LIMIT", 4)
    monkeypatch.setattr(ao, "AB_MS", 0)
    monkeypatch.setattr(ao, "stats", {k: 0 for k in ao.stats})
    monkeypatch.setattr(ao, "_state", {"engine": None, "limit": 0, "seq": 0, "window_n": 0, "charge_due": 0,
                                       "stalled_seq": -1, "in_sched": False, "busy": False, "ab_window": None})
    ao.PENDING.clear()
    ao.HELD.clear()
    yield monkeypatch
    ao.PENDING.clear()
    ao.HELD.clear()


def _install(mp, limit=4, cap=0):
    if cap:
        mp.setenv("VLLM_TPU_KV_LOAD_STARTS_PER_STEP", str(cap))
        kv_load_pacing.patch_kv_load_pacing()
    mp.setattr(ao, "LIMIT", limit)
    ao.patch_admit_off()


def _scheduler(mp, *, max_num_seqs=16, num_blocks=NUM_BLOCKS, cls=Scheduler, role="kv_consumer"):
    mp.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=2048, max_model_len=2048,
                                         enable_chunked_prefill=True, is_encoder_decoder=False),
        model_config=ModelConfig(model="facebook/opt-125m", dtype="float16", seed=42, max_model_len=2048),
        cache_config=CacheConfig(block_size=BLOCK_SIZE, gpu_memory_utilization=0.9, cache_dtype="auto",
                                 enable_prefix_caching=True),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    vllm_config.kv_transfer_config = KVTransferConfig(kv_connector="AdmitTestConnector",
                                                      kv_connector_module_path=__name__, kv_role=role)
    kv_cache_config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=64,
                                                      dtype=torch.float16))])
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    init_none_hash(sha256)
    return cls(vllm_config=vllm_config, kv_cache_config=kv_cache_config,
               structured_output_manager=StructuredOutputManager(vllm_config), block_size=BLOCK_SIZE,
               hash_block_size=BLOCK_SIZE, log_stats=True)


def _req(i, *, n=PROMPT_LEN, remote=True, num_tokens=None, unknown=False, prompt_of=None):
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    if remote:
        p = {"num_tokens": n if num_tokens is None else num_tokens}
        if unknown:
            p["unknown"] = True
        params.extra_args = {"kv_transfer_params": p}
    j = i if prompt_of is None else prompt_of
    prompt = [1000 * (j + 1) + t for t in range(n)]
    return Request("r%02d" % i, prompt, params, None, block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256))


def _engine(sched):
    e = object.__new__(EngineCoreProc)
    e.scheduler = sched
    e.input_queue = queue.Queue()
    e.shutdown_state = EngineShutdownState.RUNNING
    return e


def _drain(e):
    """The busy loop's _process_input_queue after the wait."""
    while not e.input_queue.empty():
        e._handle_client_request(*e.input_queue.get_nowait())


def _window(calls=50):
    for _ in range(calls):
        ao.step_window()


def _summary(out):
    return ([r.req_id for r in out.scheduled_new_reqs], dict(out.num_scheduled_tokens), sorted(out.finished_req_ids))


def _snapshot(s):
    pool = s.kv_cache_manager.block_pool
    return dict(
        holding=[r.request_id for r in s.kv_holding_waiting], waiting=[r.request_id for r in s.waiting],
        running=[r.request_id for r in s.running], deferred=sorted(r.request_id for r in s.deferred_waiting),
        status={rid: (r.status, r.num_computed_tokens) for rid, r in sorted(s.requests.items())},
        free=pool.get_num_free_blocks(), cached=len(pool.cached_block_hash_to_block),
        inflight=sorted(r.request_id for r in s._inflight_prefills),
    )


def _land(s, ids):
    s._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving=set(ids)))


# Arrivals, landings and finishes between passes. The prompts of r09 / r10 repeat r01 / r02's (local prefix hits).
SCRIPT = (
    ("arrive", (0, 1, 2)), ("pass",),
    ("arrive", (3,)), ("pass",),
    ("land", (0, 1)), ("pass",),
    ("arrive", (4, 5, 6, 7, 8)), ("pass",),
    ("pass",),
    ("land", (2, 3, 4)), ("finish", (0,)), ("pass",),
    ("arrive", (9, 10, 11)), ("pass",),
    ("land", (5, 6, 7, 8, 9, 10, 11)), ("pass",),
    ("finish", (1, 2)), ("arrive", (12,)), ("pass",),
    ("pass",),
)
PROMPT_OF = {9: 1, 10: 2}


def _drive(mp, lever, cls, cap):
    s = _scheduler(mp, cls=cls)
    e = _engine(s)
    if lever:
        ao.attach(e)
    rows = []
    for op in SCRIPT:
        if op[0] == "arrive":
            for i in op[1]:
                r = _req(i, prompt_of=PROMPT_OF.get(i))
                e.input_queue.put_nowait((ADD, (r, 0)))
            if not lever:
                _drain(e)
        elif op[0] == "land":
            _land(s, ["r%02d" % i for i in op[1]])
        elif op[0] == "finish":
            s.finish_requests(["r%02d" % i for i in op[1]], RequestStatus.FINISHED_STOPPED)
        else:
            if lever:
                _window()
                _drain(e)
            out = s.schedule()
            rows.append((_summary(out), _snapshot(s), list(s.connector.log)))
    return rows


@pytest.mark.parametrize("cls,cap", [(Scheduler, 0), (AsyncScheduler, 0), (Scheduler, 2), (AsyncScheduler, 4)])
def test_differential_same_passes_as_schedule(fresh, cls, cap):
    _install(fresh, limit=4, cap=cap)
    stock = _drive(fresh, False, cls, cap)
    ao.PENDING.clear()
    lever = _drive(fresh, True, cls, cap)
    assert ao.stats["admitted"] >= 8, ao.stats
    for k, (a, b) in enumerate(zip(stock, lever)):
        assert a[0] == b[0], ("pass", k, "outputs")
        assert a[1] == b[1], ("pass", k, "state")
        assert a[2] == b[2], ("pass", k, "connector")
    assert len(stock) == len(lever)


def test_admitted_request_waits_for_its_load(fresh):
    _install(fresh)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    e.input_queue.put_nowait((ADD, (_req(1), 0)))
    _window()
    r = s.requests["r01"]
    assert r.status == RequestStatus.WAITING_FOR_REMOTE_KVS and r.num_computed_tokens == PROMPT_LEN
    assert [x.request_id for x in s.kv_holding_waiting] == ["r01"] and not s.waiting
    out = s.schedule()
    assert out.scheduled_new_reqs == [] and r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert s.connector.log == [("alloc", "r01", PROMPT_LEN, s.connector.log[0][3]), ("build", ("r01",))]
    _land(s, ["r01"])
    out = s.schedule()
    assert [x.req_id for x in out.scheduled_new_reqs] == ["r01"]


def test_no_double_admission(fresh):
    _install(fresh)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    for i in range(3):
        e.input_queue.put_nowait((ADD, (_req(i), 0)))
    _window()
    for _ in range(3):
        s.schedule()
        _window()
    allocs = [x[1] for x in s.connector.log if x[0] == "alloc"]
    assert allocs == ["r00", "r01", "r02"]
    ids = [r.request_id for r in s.kv_holding_waiting] + [r.request_id for r in s.waiting]
    assert sorted(ids) == ["r00", "r01", "r02"] and len(ids) == len(set(ids))


def test_abort_of_in_window_admission_is_held_until_after_next_schedule(fresh):
    _install(fresh)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    free0 = s.kv_cache_manager.block_pool.get_num_free_blocks()
    e.input_queue.put_nowait((ADD, (_req(1), 0)))
    _window()
    assert "r01" in ao.PENDING
    assert s.finish_requests(["r01"], RequestStatus.FINISHED_ABORTED) == []
    assert s.requests["r01"].status == RequestStatus.WAITING_FOR_REMOTE_KVS and ao.HELD
    s.schedule()
    kinds = [(x[0], x[1] if x[0] != "build" else x[1]) for x in s.connector.log]
    assert kinds == [("alloc", "r01"), ("build", ("r01",)), ("finish", "r01")]
    assert s.connector.log[-1] == ("finish", "r01", False)  # the load had gone out: no leak
    assert s.requests["r01"].status == RequestStatus.FINISHED_ABORTED and not ao.HELD and not ao.PENDING
    _land(s, ["r01"])
    assert "r01" not in s.requests and s.kv_cache_manager.block_pool.get_num_free_blocks() == free0


def test_pending_clears_at_schedule(fresh):
    _install(fresh)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    e.input_queue.put_nowait((ADD, (_req(1), 0)))
    _window()
    s.schedule()
    assert not ao.PENDING
    s.finish_requests(["r01"], RequestStatus.FINISHED_ABORTED)
    assert not ao.HELD and s.requests["r01"].status == RequestStatus.FINISHED_ABORTED


def test_pull_takes_only_a_head_add(fresh):
    _install(fresh)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    e.input_queue.put_nowait((ABORT, ["rxx"]))
    e.input_queue.put_nowait((ADD, (_req(1), 0)))
    _window()
    assert ao.stats["pulled"] == 0 and e.input_queue.qsize() == 2 and "r01" not in s.requests
    s.schedule()
    _drain(e)
    e.input_queue.put_nowait((ADD, (_req(2), 0)))
    e.input_queue.put_nowait((ABORT, ["r02"]))
    e.input_queue.put_nowait((ADD, (_req(3), 0)))
    _window()
    assert ao.stats["pulled"] == 1 and "r02" in s.requests and "r03" not in s.requests
    assert [x[0] for x in e.input_queue.queue] == [ABORT, ADD]


def test_backout_when_not_async_leaves_no_trace(fresh):
    _install(fresh)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    # r01 runs and finishes: its prompt's full blocks are cached.
    e.input_queue.put_nowait((ADD, (_req(1), 0)))
    _window()
    s.schedule()
    _land(s, ["r01"])
    s.schedule()
    s.finish_requests(["r01"], RequestStatus.FINISHED_STOPPED)
    s.schedule()
    # r02 repeats the prompt; its producer published only the 32 tokens the local hit covers: nothing to load.
    r = _req(2, prompt_of=1, num_tokens=32)
    before = _snapshot(s)
    log0 = len(s.connector.log)
    e.input_queue.put_nowait((ADD, (r, 0)))
    _window()
    after = _snapshot(s)
    assert ao.stats["backout"] == 1 and "r02" not in ao.PENDING
    assert after["waiting"] == ["r02"] and after["holding"] == before["holding"]
    assert after["free"] == before["free"] and after["cached"] == before["cached"]
    assert r.status == RequestStatus.WAITING and r.num_computed_tokens == 0 and r not in s._inflight_prefills
    assert not [x for x in s.connector.log[log0:] if x[0] == "alloc"]


def test_backout_when_blocks_do_not_fit_leaves_no_trace(fresh):
    _install(fresh)
    s = _scheduler(fresh, num_blocks=20)
    e = _engine(s)
    ao.attach(e)
    r = _req(1, n=400)
    free0 = s.kv_cache_manager.block_pool.get_num_free_blocks()
    e.input_queue.put_nowait((ADD, (r, 0)))
    _window()
    assert ao.stats["backout"] == 1 and ao.stats["admitted"] == 0
    assert [x.request_id for x in s.waiting] == ["r01"] and not s.kv_holding_waiting
    assert r.status == RequestStatus.WAITING and r.num_computed_tokens == 0
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == free0 and not s.connector.log


def test_backout_when_the_connector_cannot_match(fresh):
    _install(fresh)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    r = _req(1, unknown=True)
    e.input_queue.put_nowait((ADD, (r, 0)))
    _window()
    assert ao.stats["backout"] == 1 and [x.request_id for x in s.waiting] == ["r01"] and not s.connector.log


def test_in_window_starts_are_charged_to_the_next_pass(fresh):
    _install(fresh, limit=8, cap=2)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    assert ao._state["limit"] == 2
    for i in range(5):
        e.input_queue.put_nowait((ADD, (_req(i), 0)))
    _window()
    _drain(e)
    assert ao.stats["admitted"] == 2
    s.schedule()
    allocs = [x[1] for x in s.connector.log if x[0] == "alloc"]
    assert allocs == ["r00", "r01"], allocs
    assert ao.stats["charged"] == 2
    _window()
    s.schedule()
    allocs = [x[1] for x in s.connector.log if x[0] == "alloc"]
    assert allocs == ["r00", "r01", "r02", "r03"], allocs


def test_limit_caps_admissions_between_schedules(fresh):
    _install(fresh, limit=3)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    for i in range(5):
        e.input_queue.put_nowait((ADD, (_req(i), 0)))
    _window()
    assert ao.stats["admitted"] == 3 and e.input_queue.qsize() == 2


def test_ineligible_requests_are_left_to_schedule(fresh):
    _install(fresh)
    # No free running slot: the next pass could not start a load either.
    s = _scheduler(fresh, max_num_seqs=1)
    e = _engine(s)
    ao.attach(e)
    s.add_request(_req(0, remote=False))
    s.schedule()
    assert len(s.running) == 1
    e.input_queue.put_nowait((ADD, (_req(1), 0)))
    _window()
    assert ao.stats["admitted"] == 0 and ao.stats["ineligible"] == 1
    assert not [x for x in s.connector.log if x[0] == "alloc"]
    assert [x.request_id for x in s.waiting] == ["r01"]
    # A request without remote KV is not admitted in the wait.
    s2 = _scheduler(fresh)
    e2 = _engine(s2)
    ao.attach(e2)
    ao._state["stalled_seq"] = -1
    e2.input_queue.put_nowait((ADD, (_req(2, remote=False), 0)))
    _window()
    assert [x.request_id for x in s2.waiting] == ["r02"] and s2.requests["r02"].status == RequestStatus.WAITING


def test_engage_refuses_a_producer(fresh, monkeypatch):
    lines = []
    monkeypatch.setattr(ao.logger, "warning", lambda fmt, *a: lines.append(fmt % a))
    _install(fresh)
    s = _scheduler(fresh, role="kv_producer")
    ao.attach(_engine(s))
    assert ao._state["engine"] is None
    assert lines == ["ADMIT_OFF not engaged: not a KV consumer; admissions run in schedule() as before"], lines


class _Cond:
    pass


def test_only_long_main_thread_waits_admit(fresh, monkeypatch):
    _install(fresh)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    e.input_queue.put_nowait((ADD, (_req(1), 0)))
    clock = [10_000_000]
    monkeypatch.setattr(ao.time, "monotonic_ns", lambda: clock[0])
    c = _Cond()
    ao.note_wait(c)  # first wait() call of this wait: starts the clock
    clock[0] += ao.OPEN_AFTER_NS - 1
    ao.note_wait(c)
    assert ao.stats["pulled"] == 0
    # Another thread's wait never works, however long.
    t = threading.Thread(target=lambda: [ao.note_wait(c) for _ in range(3)])
    clock[0] += 10 * ao.OPEN_AFTER_NS
    t.start()
    t.join()
    assert ao.stats["pulled"] == 0
    ao.note_read(c)  # the read returned: the next wait starts afresh
    ao.note_wait(c)
    assert ao.stats["pulled"] == 0
    clock[0] += ao.OPEN_AFTER_NS
    ao.note_wait(c)
    assert ao.stats["pulled"] == 1 and ao.stats["admitted"] == 1


def test_ab_windows_follow_the_wall_clock(fresh, monkeypatch):
    _install(fresh)
    monkeypatch.setattr(ao, "AB_MS", 1000)
    s = _scheduler(fresh)
    e = _engine(s)
    ao.attach(e)
    now = [7 * 1_000_000_000]  # odd window: off
    monkeypatch.setattr(ao.time, "time_ns", lambda: now[0])
    e.input_queue.put_nowait((ADD, (_req(1), 0)))
    _window()
    assert ao.stats["admitted"] == 0 and ao.stats["pulled"] == 0 and ao.stats["ab_off"] > 0
    assert ao.ab_on(7 * 10**9) is False and ao.ab_on(8 * 10**9) is True
    now[0] = 8 * 1_000_000_000  # even window: on
    _window()
    assert ao.stats["admitted"] == 1


def test_key_off_patches_nothing(fresh, monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    assert envs.VLLM_TPU_ADMIT_OFF == 0 and envs.VLLM_TPU_ADMIT_OFF_AB_MS == 0
    monkeypatch.setattr(ao, "LIMIT", 0)
    before = (Scheduler.schedule, Scheduler._handle_blocked_waiting_request, Scheduler.finish_requests,
              SpinCondition.wait, SpinCondition.record_read, EngineCoreProc.__init__)
    ao.patch_admit_off()
    after = (Scheduler.schedule, Scheduler._handle_blocked_waiting_request, Scheduler.finish_requests,
             SpinCondition.wait, SpinCondition.record_read, EngineCoreProc.__init__)
    assert all(x is y for x, y in zip(before, after))


def test_key_on_installs_once_and_refuses_the_spinloop_extension(fresh, monkeypatch):
    orig = (Scheduler.schedule, Scheduler._handle_blocked_waiting_request, Scheduler.finish_requests)
    monkeypatch.setattr(shm_broadcast, "SPINLOOP_EXT_ENABLED", True, raising=False)
    ao.patch_admit_off()
    assert (Scheduler.schedule, Scheduler._handle_blocked_waiting_request, Scheduler.finish_requests) == orig
    monkeypatch.setattr(shm_broadcast, "SPINLOOP_EXT_ENABLED", False, raising=False)
    ao.patch_admit_off()
    ao.patch_admit_off()
    for fn in (Scheduler.schedule, Scheduler._handle_blocked_waiting_request, Scheduler.finish_requests,
               SpinCondition.wait, SpinCondition.record_read, EngineCoreProc.__init__):
        assert getattr(fn, ao._PATCH_FLAG, False)
    assert (Scheduler.schedule.__wrapped__, Scheduler._handle_blocked_waiting_request.__wrapped__,
            Scheduler.finish_requests.__wrapped__) == orig


def test_source_fingerprints_match_and_a_mismatch_refuses(fresh, monkeypatch):
    assert ao.check_fingerprints() == []
    monkeypatch.setitem(ao._FINGERPRINTS, "Scheduler.schedule", "0" * 16)
    orig = Scheduler.schedule
    ao.patch_admit_off()
    assert Scheduler.schedule is orig


def test_engine_core_logs_the_engagement_line(fresh, monkeypatch):
    lines = []
    monkeypatch.setattr(ao.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    s = _scheduler(fresh)

    def stub_init(self, *a, **k):
        self.scheduler = s
        self.input_queue = queue.Queue()
        self.shutdown_state = EngineShutdownState.RUNNING

    monkeypatch.setattr(EngineCoreProc, "__init__", stub_init)
    ao.patch_admit_off()
    lines.clear()
    eng = object.__new__(EngineCoreProc)
    EngineCoreProc.__init__(eng)
    assert lines == ["ADMIT_OFF engaged: remote-KV admissions run in the engine core's token wait "
                     "(up to 4 between schedules, after 2.0 ms of a wait; KV load pacing cap 0; AB 0 ms)"], lines
    assert ao._state["engine"] is eng


def test_keys_are_registered_and_out_of_the_compile_key():
    import ast
    import pathlib

    assert all(k in envs.environment_variables for k in KEYS)
    src = pathlib.Path(ao.__file__).parents[1] / "compilation" / "tpu_compiler.py"
    ignored = None
    for node in ast.walk(ast.parse(src.read_text())):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_TPU_COMPILE_ENV_IGNORED" for t in node.targets
        ):
            ignored = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    assert ignored is not None and all(k in ignored for k in KEYS)


def test_registry_installs_it_at_engine_core_after_pacing_and_defer_fin():
    from vllm_torchtpu import patch_registry

    targets = [p.target for p in patch_registry.PATCHES]
    t = "vllm_torchtpu.core.admit_off:patch_admit_off"
    assert patch_registry.PATCHES[targets.index(t)].stages == ("engine_core",)
    for earlier in ("vllm_torchtpu.core.kv_load_pacing:patch_kv_load_pacing",
                    "vllm_torchtpu.core.kv_decode_fast_path:patch_kv_decode_fast_path",
                    "vllm_torchtpu.core.defer_fin:patch_defer_fin"):
        assert targets.index(t) > targets.index(earlier)


STACK = r"""
import os, sys
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm_torchtpu import patch_registry
from vllm_torchtpu.core import kv_decode_fast_path as fp
WANT = ("vllm_torchtpu:_patch_vllm_same_step_prefix_hits", "vllm_torchtpu.core.kv_decode_fast_path:",
        "vllm_torchtpu.core.kv_load_pacing:", "vllm_torchtpu.core.preproc_in_wait:", "vllm_torchtpu.core.defer_fin:",
        "vllm_torchtpu.core.admit_off:")
names = ("allocate_slots", "cache_blocks", "free", "pop_blocks_for_free", "remove_skipped_blocks")
snap = None
before = None
order = []
for p in patch_registry.PATCHES:
    if not p.target.startswith(WANT):
        continue
    if p.target.startswith("vllm_torchtpu.core.kv_decode_fast_path:"):
        before = fp.check_fingerprints()
    patch_registry._invoke(p.target)
    order.append(p.target.split(":")[-1])
    if p.target.startswith("vllm_torchtpu.core.kv_decode_fast_path:"):
        snap = tuple(KVCacheManager.__dict__[n] for n in names)
from vllm_torchtpu.core import host_knobs
host_knobs.patch_engine_core_for_host_knobs()
order.append("host_knobs")
same = snap is not None and snap == tuple(KVCacheManager.__dict__[n] for n in names)
flag = bool(getattr(KVCacheManager.allocate_slots, fp._PATCH_FLAG, False))
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_torchtpu.core import admit_off as ao
# check_fingerprints() hashes the installed KVCacheManager attributes, so after the fast path installs it lists its own
# wrappers; what matters is that it was empty when the fast path installed and that nothing replaced them since.
print("STACK order=%s fp_mismatch_at_install=%s fast_path_flag=%d kvm_unchanged=%d admit_off=%d fp_after=%s" % (
    ",".join(order), before, flag, same, int(getattr(Scheduler.finish_requests, ao._PATCH_FLAG, False)),
    fp.check_fingerprints()), flush=True)
"""


def _stack_env(admit):
    env = dict(os.environ, VLLM_TPU_KV_DECODE_FAST_PATH="1", VLLM_TPU_KV_LOAD_STARTS_PER_STEP="4",
               VLLM_TPU_PREPROC_IN_WAIT="30", VLLM_TPU_DEFER_FIN="1", VLLM_TPU_HOST_SWITCH_US="200",
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("VLLM_TPU_ADMIT_OFF", None)
    if admit:
        env["VLLM_TPU_ADMIT_OFF"] = "4"
    return env


@pytest.mark.parametrize("admit", [True, False])
def test_car_h_stack_keeps_the_fast_path(admit):
    r = subprocess.run([sys.executable, "-c", STACK], env=_stack_env(admit), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    line = [x for x in r.stdout.splitlines() if x.startswith("STACK ")]
    assert r.returncode == 0 and len(line) == 1, r.stdout[-3000:]
    assert "fp_mismatch_at_install=[] fast_path_flag=1 kvm_unchanged=1 admit_off=%d " % int(admit) in line[0], line[0]
