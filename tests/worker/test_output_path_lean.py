# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_OUTPUT_PATH_LEAN: the engine must read the same responses, in the
same order, with the same bytes on the reply ring, as with vLLM's own
``WorkerProc`` output path.

Uses vLLM's real ``WorkerProc`` methods (on a copy of the class, so the real
class stays untouched), real ``MessageQueue`` rings in shared memory, and a
fake worker whose ``sample_tokens`` outputs wait a random time for their
"device copy", so the output queue backs up sometimes and drains other times.
"""

import contextlib
import copyreg
import pickle
import queue
import random
import time
import types
from threading import Thread

import pytest
import torch
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.worker import output_path_lean as lean

_METHODS = ("enqueue_output", "handle_output", "async_output_busy_loop")
SUCCESS = WorkerProc.ResponseStatus.SUCCESS


def _clone_worker_proc() -> type:
    """A copy of vLLM's WorkerProc with the same function objects."""
    ns = {
        k: v
        for k, v in WorkerProc.__dict__.items()
        if k not in ("__dict__", "__weakref__")
    }
    return type("WorkerProcClone", (), ns)


# --------------------------------------------------------------------------
# rings
# --------------------------------------------------------------------------


def _ring(max_chunk_bytes=1 << 20, max_chunks=64):
    w = MessageQueue(1, 1, max_chunk_bytes=max_chunk_bytes, max_chunks=max_chunks)
    r = MessageQueue.create_from_handle(w.export_handle(), 0)
    return w, r


class _Socket:
    """Records the overflow path's multipart sends."""

    def __init__(self):
        self.sent = []

    def send_multipart(self, bufs, copy=False):
        self.sent.append([bytes(b) for b in bufs])


def _shm_bytes(mq) -> bytes:
    return bytes(mq.buffer.shared_memory.buf)


# --------------------------------------------------------------------------
# fake worker and step script
# --------------------------------------------------------------------------


class _StopLoop(BaseException):
    pass


class _Queue(queue.Queue):
    """Raises in the output thread on the sentinel so the loop ends."""

    def get(self, *a, **k):
        item = super().get(*a, **k)
        if item is _STOP:
            raise _StopLoop
        return item


_STOP = object()


class _Async(AsyncModelRunnerOutput):
    def __init__(self, result, delay_s):
        self._result = result
        self._delay_s = delay_s

    def get_output(self):
        time.sleep(self._delay_s)  # the D2H copy
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _mro(step, n, finished=()):
    ids = [f"req-{i}" for i in range(n) if i not in finished]
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=[
            [step * 1000 + i] * (1 + (i + step) % 3) for i in range(len(ids))
        ],
    )


def _script(seed, steps=120):
    """(method, result, delay_s, pause_s) per RPC. Results are fixed by the
    seed; only timings vary between runs."""
    rng = random.Random(seed)
    out = []
    n = 16
    for step in range(steps):
        kind = rng.random()
        pause = rng.choice([0.0, 0.0, 0.0005, 0.004])
        if kind < 0.05:
            # execute_model raises (WorkerProc sends FAILURE).
            out.append(("execute_model", ValueError(f"boom {step}"), 0.0, pause))
            continue
        if kind < 0.12:
            # Empty step: execute_model returns an output, no sample_tokens.
            out.append(("execute_model", EMPTY_MODEL_RUNNER_OUTPUT, 0.0, pause))
            continue
        out.append(("execute_model", None, 0.0, pause))
        finished = tuple(rng.sample(range(n), rng.randrange(0, 3)))  # finished/aborted
        if kind < 0.17:
            res = RuntimeError(f"d2h failed {step}")  # get_output raises
        else:
            res = _mro(step, n, finished)
        delay = rng.choice([0.0, 0.0002, 0.001, 0.003])
        out.append(
            ("sample_tokens", _Async(res, delay), 0.0, rng.choice([0.0, 0.0, 0.004]))
        )
        if rng.random() < 0.1:
            # Another RPC with an output rank that returns None.
            out.append(("take_draft_token_ids", None, 0.0, 0.0))
        if kind > 0.97:
            # An RPC with no output rank (all ranks reply).
            out.append(("all_ranks", {"step": step}, 0.0, 0.0))
    return out


class _FakeWorker:
    def __init__(self):
        self.next = None

    def _ret(self, *a, **k):
        r = self.next
        if isinstance(r, Exception):
            raise r
        return r

    execute_model = sample_tokens = take_draft_token_ids = all_ranks = _ret


def _make_proc(cls, mq, async_sched=True):
    self = cls.__new__(cls)
    self.rank = 0
    self.worker = _FakeWorker()
    self.use_async_scheduling = async_sched
    self.worker_response_mq = mq
    if async_sched:
        self.async_output_queue = _Queue()
        # Bound now, as vLLM binds Thread(target=self.async_output_busy_loop).
        loop = self.async_output_busy_loop

        def run():
            with contextlib.suppress(_StopLoop):
                loop()

        self.async_output_copy_thread = Thread(
            target=run, daemon=True, name="WorkerAsyncOutputCopy"
        )
        self.async_output_copy_thread.start()
    return self


def _run_script(cls, script, async_sched=True):
    """Drives vLLM's real _execute_worker_rpc; returns the responses the
    engine reads, as (status, pickled result) pairs, plus the raw frames."""
    w, r = _ring()
    got = []

    def reader():
        while True:
            try:
                status, result = r.dequeue(timeout=2.0)
            except TimeoutError:
                return
            got.append((status, pickle.dumps(result, protocol=5)))
            if len(got) == expected:
                return

    expected = sum(1 for m, *_ in script)
    rt = Thread(target=reader, daemon=True)
    rt.start()
    proc = _make_proc(cls, w, async_sched)
    for method, result, _, pause in script:
        proc.worker.next = result
        output_rank = None if method == "all_ranks" else 0
        proc._execute_worker_rpc((method, (), {}, output_rank))
        if pause:
            time.sleep(pause)
    rt.join(timeout=30)
    if async_sched:
        proc.async_output_queue.put(_STOP)
        proc.async_output_copy_thread.join(timeout=5)
    return got, proc


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("lean_mq", [True, False])
def test_same_response_sequence(seed, lean_mq):
    script = _script(seed)
    base, _ = _run_script(_clone_worker_proc(), script)
    patched = _clone_worker_proc()
    assert lean.wrap_worker_proc(patched, use_lean_mq=lean_mq)
    got, proc = _run_script(patched, script)
    assert len(base) == len(script)
    assert got == base
    st = proc.__dict__[lean._STATE]
    # Both paths ran: None sent inline (thread idle) and queued (backlog).
    assert st.inline > 0
    assert st.queued_none > 0
    assert st.puts == st.done
    if lean_mq:
        w = proc.worker_response_mq.__dict__[lean._MQ_STATE]
        assert w.fast > 0


def test_same_response_sequence_sync_scheduling():
    script = [s for s in _script(5, steps=40)]
    base, _ = _run_script(_clone_worker_proc(), script, async_sched=False)
    patched = _clone_worker_proc()
    assert lean.wrap_worker_proc(patched, use_lean_mq=True)
    got, _ = _run_script(patched, script, async_sched=False)
    assert got == base and len(got) == len(script)


def test_original_loop_never_goes_inline():
    """Patch applied after the thread started vLLM's own loop: loop_ok stays
    False, so every item is queued (vLLM's order trivially)."""
    cls = _clone_worker_proc()
    w, r = _ring()
    proc = _make_proc(cls, w)  # thread runs the original loop
    assert lean.wrap_worker_proc(cls, use_lean_mq=True)
    for _ in range(5):
        proc.worker.next = None
        proc._execute_worker_rpc(("execute_model", (), {}, 0))
    for _ in range(5):
        assert r.dequeue(timeout=5) == (SUCCESS, None)
    st = proc.__dict__[lean._STATE]
    assert st.inline == 0 and st.queued_none == 5
    proc.async_output_queue.put(_STOP)


# --------------------------------------------------------------------------
# (2) lean writer: bytes on the ring
# --------------------------------------------------------------------------


class _Custom:
    def __init__(self, v):
        self.v = v


def _payloads():
    big_list = list(range(60_000))  # ~300 KB pickle > _KEEP_BYTES
    return [
        ("none", (SUCCESS, None)),
        ("mro", (SUCCESS, _mro(3, 16))),
        ("long", (SUCCESS, big_list)),
        ("short", (SUCCESS, [1])),
        ("none2", (SUCCESS, None)),
        ("long2", (SUCCESS, big_list[::-1])),
        ("tensor_inband", (SUCCESS, torch.arange(1000, dtype=torch.int32))),
        ("tensor_oob", (SUCCESS, torch.ones(300_000, dtype=torch.float32))),  # 1.2 MB
        ("overflow", (SUCCESS, b"x" * (3 << 20))),  # > chunk
        ("short2", (SUCCESS, "ab")),
        ("failure", (WorkerProc.ResponseStatus.FAILURE, "err")),
        ("none3", (SUCCESS, None)),
    ]


def _eq(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b


def test_lean_enqueue_bytes_identical_and_no_bleed():
    a_w, a_r = _ring(max_chunk_bytes=2 << 20, max_chunks=4)
    b_w, b_r = _ring(max_chunk_bytes=2 << 20, max_chunks=4)
    a_w.local_socket = _Socket()
    b_w.local_socket = _Socket()
    for rep in range(3):  # long -> short -> long, cycling the 4 ring slots
        for name, obj in _payloads():
            a_w.enqueue(obj)
            lean.lean_enqueue(b_w, obj, none_status=SUCCESS if obj[1] is None else None)
            assert _shm_bytes(a_w) == _shm_bytes(b_w), (rep, name)
            assert a_w.local_socket.sent == b_w.local_socket.sent, (rep, name)
            assert a_w.current_idx == b_w.current_idx
            if name != "overflow":  # overflow payload went to the fake socket
                ra, rb = a_r.dequeue(timeout=5), b_r.dequeue(timeout=5)
                assert _eq(ra, obj) and _eq(rb, obj), (rep, name)
            else:
                # Readers would take the socket; skip the slot's overflow flag.
                for r in (a_r, b_r):
                    with r.acquire_read(timeout=5) as buf:
                        assert buf[0] == 1
    st = b_w.__dict__[lean._MQ_STATE]
    # tensor_oob and overflow go through the original enqueue.
    assert st.fallback == 3 * 2
    assert st.fast == 3 * (len(_payloads()) - 2)
    # The 300 KB buffer is not kept after the last long payload.
    assert st.bio is None or st.bio.tell() <= lean._KEEP_BYTES


def test_none_frame_matches_fresh_pickle():
    w, r = _ring(max_chunk_bytes=1 << 16, max_chunks=4)
    lean.lean_enqueue(w, (SUCCESS, None), none_status=SUCCESS)
    lean.lean_enqueue(w, (SUCCESS, None), none_status=SUCCESS)  # from cache
    frame = w.__dict__[lean._MQ_STATE].none_frame
    payload = pickle.dumps((SUCCESS, None), protocol=pickle.HIGHEST_PROTOCOL)
    assert frame == b"\x00\x00\x01" + len(payload).to_bytes(4, "big") + payload
    assert r.dequeue(timeout=5) == (SUCCESS, None)
    assert r.dequeue(timeout=5) == (SUCCESS, None)


def test_dispatch_table_change_rebuilds_pickler(monkeypatch):
    a_w, a_r = _ring(max_chunk_bytes=1 << 16, max_chunks=4)
    b_w, b_r = _ring(max_chunk_bytes=1 << 16, max_chunks=4)
    obj = (SUCCESS, _Custom(7))
    for step in range(2):
        if step == 1:
            # A reducer registered after the first send must be honoured.
            monkeypatch.setitem(
                copyreg.dispatch_table, _Custom, lambda c: (_Custom, (c.v + 1,))
            )
        a_w.enqueue(obj)
        lean.lean_enqueue(b_w, obj)
        assert _shm_bytes(a_w) == _shm_bytes(b_w)
        assert a_r.dequeue(timeout=5)[1].v == b_r.dequeue(timeout=5)[1].v == 7 + step


def test_full_slot_falls_back_and_times_out_like_original():
    a_w, _ = _ring(max_chunk_bytes=1 << 16, max_chunks=2)
    b_w, _ = _ring(max_chunk_bytes=1 << 16, max_chunks=2)
    for i in range(2):  # fill both slots, nobody reads
        a_w.enqueue((SUCCESS, i))
        lean.lean_enqueue(b_w, (SUCCESS, i))
    with pytest.raises(TimeoutError):
        a_w.enqueue((SUCCESS, 9), timeout=0.05)
    with pytest.raises(TimeoutError):
        lean.lean_enqueue(b_w, (SUCCESS, 9), timeout=0.05)
    assert _shm_bytes(a_w) == _shm_bytes(b_w)
    assert b_w.__dict__[lean._MQ_STATE].fallback == 1


def test_concurrent_caller_uses_original(monkeypatch):
    w, r = _ring(max_chunk_bytes=1 << 16, max_chunks=4)
    lean.lean_enqueue(w, (SUCCESS, 1))
    st = w.__dict__[lean._MQ_STATE]
    calls = []
    orig = MessageQueue.enqueue
    monkeypatch.setattr(
        MessageQueue,
        "enqueue",
        lambda self, o, t=None: (calls.append(o), orig(self, o, t)),
    )
    assert st.lock.acquire()
    try:
        lean.lean_enqueue(w, (SUCCESS, 2))
    finally:
        st.lock.release()
    assert calls == [(SUCCESS, 2)]
    assert r.dequeue(timeout=5) == (SUCCESS, 1)
    assert r.dequeue(timeout=5) == (SUCCESS, 2)


def test_unsupported_queue_uses_original(monkeypatch):
    w, _ = _ring(max_chunk_bytes=1 << 16, max_chunks=4)
    w.tensor_arena = object()  # v0.32 arena: never the fast path
    calls = []
    monkeypatch.setattr(
        MessageQueue, "enqueue", lambda self, o, t=None: calls.append(o)
    )
    lean.lean_enqueue(w, (SUCCESS, 3))
    assert calls == [(SUCCESS, 3)]
    assert w.__dict__[lean._MQ_STATE] is False


# --------------------------------------------------------------------------
# install / flag
# --------------------------------------------------------------------------


def test_sources_match_this_vllm():
    assert lean.worker_sources_match(WorkerProc)
    assert lean.mq_sources_match()


def test_flag_off_installs_nothing(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_OUTPUT_PATH_LEAN", raising=False)
    monkeypatch.delitem(envs.__dict__, "VLLM_TPU_OUTPUT_PATH_LEAN", raising=False)
    before = {m: WorkerProc.__dict__[m] for m in _METHODS}
    mq_before = MessageQueue.__dict__["enqueue"]
    lean.patch_output_path_lean()
    assert {m: WorkerProc.__dict__[m] for m in _METHODS} == before
    assert all(WorkerProc.__dict__[m] is before[m] for m in _METHODS)
    assert MessageQueue.__dict__["enqueue"] is mq_before
    assert not getattr(WorkerProc, lean._MARK, False)


def test_shape_mismatch_installs_nothing():
    cls = _clone_worker_proc()

    def handle_output(self, output):  # a different vLLM version
        self.enqueue_output(output)

    cls.handle_output = handle_output
    before = {m: cls.__dict__[m] for m in _METHODS}
    assert not lean.wrap_worker_proc(cls, use_lean_mq=True)
    assert {m: cls.__dict__[m] for m in _METHODS} == before


def test_env_registered_and_out_of_compile_key(monkeypatch):
    assert "VLLM_TPU_OUTPUT_PATH_LEAN" in envs.environment_variables
    assert "VLLM_TPU_OUTPUT_PATH_LEAN" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv("VLLM_TPU_OUTPUT_PATH_LEAN", raising=False)
    assert envs.environment_variables["VLLM_TPU_OUTPUT_PATH_LEAN"]() is False
    monkeypatch.setenv("VLLM_TPU_OUTPUT_PATH_LEAN", "1")
    assert envs.environment_variables["VLLM_TPU_OUTPUT_PATH_LEAN"]() is True


def test_registered_in_patch_registry():
    from vllm_torchtpu import patch_registry

    targets = {p.target: p.stages for p in patch_registry.PATCHES}
    t = "vllm_torchtpu.worker.output_path_lean:patch_output_path_lean"
    assert t in targets
    assert "worker_init" in targets[t]


# --------------------------------------------------------------------------
# smoke-construct: the patch installs on the real classes and the objects
# are built the way WorkerProc.__init__ / _init_message_queues build them.
# --------------------------------------------------------------------------


def test_smoke_construct_real_worker_proc(monkeypatch):
    import importlib

    from vllm_torchtpu import patch_registry

    saved = {m: WorkerProc.__dict__[m] for m in _METHODS}
    monkeypatch.setenv("VLLM_TPU_OUTPUT_PATH_LEAN", "1")
    monkeypatch.delitem(envs.__dict__, "VLLM_TPU_OUTPUT_PATH_LEAN", raising=False)
    (entry,) = [p for p in patch_registry.PATCHES if "output_path_lean" in p.target]
    mod_name, fn_name = entry.target.split(":")
    try:
        getattr(importlib.import_module(mod_name), fn_name)()
        assert getattr(WorkerProc, lean._MARK, False)
        assert all(WorkerProc.__dict__[m] is not saved[m] for m in _METHODS)

        # Engine side: the broadcast ring the executor creates.
        engine_rpc_mq = MessageQueue(1, 1)
        self = WorkerProc.__new__(WorkerProc)
        self.rank = 0
        self.worker = types.SimpleNamespace(rank=0)
        cfg = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(nnodes_within_dp=1)
        )
        # vLLM's own _init_message_queues builds the reply ring.
        WorkerProc._init_message_queues(self, engine_rpc_mq.export_handle(), cfg)
        # As WorkerProc.__init__ with async scheduling.
        self.use_async_scheduling = True
        self.async_output_queue = queue.Queue()
        self.async_output_copy_thread = Thread(
            target=self.async_output_busy_loop,
            daemon=True,
            name="WorkerAsyncOutputCopy",
        )
        self.async_output_copy_thread.start()
        # The executor's reader of this rank's replies, with the handshake.
        reader = MessageQueue.create_from_handle(
            self.worker_response_mq.export_handle(), 0
        )
        t = Thread(target=self.worker_response_mq.wait_until_ready)
        t.start()
        reader.wait_until_ready()
        t.join(timeout=10)

        self.worker = _FakeWorker()
        script = [
            ("execute_model", None),
            ("sample_tokens", _Async(_mro(1, 4), 0.002)),
            ("execute_model", None),
            ("sample_tokens", _Async(_mro(2, 4), 0.0)),
        ]
        for method, res in script:
            self.worker.next = res
            self._execute_worker_rpc((method, (), {}, 0))
        got = [reader.dequeue(timeout=10) for _ in script]
        assert got == [
            (SUCCESS, None),
            (SUCCESS, _mro(1, 4)),
            (SUCCESS, None),
            (SUCCESS, _mro(2, 4)),
        ]
        st = self.__dict__[lean._STATE]
        assert st.loop_ok and st.inline >= 1
        assert self.worker_response_mq.__dict__[lean._MQ_STATE].fast >= 3
    finally:
        for m, fn in saved.items():
            setattr(WorkerProc, m, fn)
        if lean._MARK in WorkerProc.__dict__:
            delattr(WorkerProc, lean._MARK)
