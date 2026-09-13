"""Unit tests for ZmqShmKvConnectorBase and its module-level helpers.

Validates the KV-transfer coordination protocol, shared-memory staging
lifecycle, PULL serving paths, and background event loops.

Hardware dependencies (TPU devices, POSIX shared memory, ZeroMQ sockets) are
substituted with lightweight in-memory test doubles:

  * `_FakeConnector` implements abstract transport hooks with CPU tensors to
    validate staging and scatter bookkeeping without hardware accelerators.
  * `_RecordingPool` subclasses `HostKVShmPool` on a temporary shm segment to
    record connector interactions while verifying production byte layouts.
  * `_FakeSocket` replays scripted inbound frames and records sent messages,
    returning 0 on poll() upon script exhaustion to terminate listener loops.

Event loops governed by timeouts substitute `_stop_event` with `_ScriptedEvent`
to execute deterministic, single-sweep passes without sleeping on the clock.
"""
import hashlib
import itertools
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import zmq

from vllm_torchtpu.distributed.kv_transfer import zmq_shm_base as zsb
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    LoadMeta, SendMeta, TPUConnectorMetadata)
from vllm_torchtpu.distributed.kv_transfer.host_kv_shm import (HostKVShmPool,
                                                               PoolSpec)

_BASE = "vllm_torchtpu.distributed.kv_transfer.zmq_shm_base"

_NUM_LAYERS = 2
_CACHE_BLOCKS = 16
_BLOCK_SHAPE = (4, 2)
_DTYPE = torch.float32
_ELEM_BYTES = 4

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _ScriptedEvent(threading.Event):
    """Threading event that yields a scripted sequence of wait() results.

    Used as `_stop_event` in loops that block on `wait(timeout)`: each False
    result executes one iteration of the loop body, and the first True sets
    the event to terminate the loop deterministically.
    """

    def __init__(self, wait_results):
        super().__init__()
        self._results = list(wait_results)

    def wait(self, timeout=None):  # type: ignore[override]
        if self._results:
            result = self._results.pop(0)
        else:
            result = True
        if result:
            self.set()
        return result


class _VanishingDict(dict):
    """Dictionary subclass whose pop() always raises KeyError to simulate deletion.

    Models the race window in the expiration sweeper between scanning
    `_coord_send.items()` and popping the UUID when `get_finished()`
    concurrently deletes the entry from the main thread.
    """

    def pop(self, key, default=None):
        super().pop(key, default)
        return default


class _FakeSocket:
    """Scripted ZeroMQ socket test double that replays frames and logs sends.

    Any Exception instance present in `inbound` is raised on recv_multipart().
    """

    def __init__(self,
                 inbound=None,
                 on_exhausted=None,
                 send_error=None,
                 empty_polls=0):
        self.inbound = list(inbound or [])
        self.on_exhausted = on_exhausted
        self.send_error = send_error
        # Number of initial poll() calls that return 0 (not readable) before
        # yielding scripted frames.
        self.empty_polls = empty_polls
        self.sent: list = []
        self.opts: dict = {}
        self.bound = None
        self.connected = None
        self.closed = False
        self.polls = 0

    def setsockopt(self, opt, value):
        self.opts[opt] = value

    def bind(self, path):
        self.bound = path

    def connect(self, path):
        self.connected = path

    def poll(self, timeout=0):
        self.polls += 1
        if self.empty_polls > 0:
            self.empty_polls -= 1
            return 0
        if self.inbound:
            return 1
        if self.on_exhausted is not None:
            self.on_exhausted()
        return 0

    def recv_multipart(self, copy=True):
        item = self.inbound.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def send_multipart(self, frames, copy=True):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(list(frames))

    def send_string(self, text):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(text)

    def close(self, linger=0):
        self.closed = True


class _FakeContext:

    def __init__(self):
        self.sockets: list[_FakeSocket] = []
        self.destroyed = False

    def socket(self, socket_type):
        sock = _FakeSocket()
        sock.socket_type = socket_type
        self.sockets.append(sock)
        return sock

    def destroy(self, linger=0):
        self.destroyed = True


class _RecordingPool(HostKVShmPool):
    """HostKVShmPool subclass that records invocations while preserving real shm operations.

    Subclassing the production pool ensures end-to-end byte layout fidelity:
    slot allocation, tensor views passed to H2D, memoryviews passed to ZeroMQ
    zero-copy sends, and per-layer size validation all execute against real
    shared memory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acquired: list[int] = []
        self.released: list[int] = []
        self.unpacked: list[tuple] = []
        self.mlocked_rank = None
        self.closed = False

    def acquire_slot(self, timeout=None):
        slot = super().acquire_slot(timeout=timeout)
        self.acquired.append(slot)
        return slot

    def release_slot(self, slot_idx):
        self.released.append(slot_idx)
        super().release_slot(slot_idx)

    def unpack_rank_layers(self, slot_idx, rank, num_blocks, layer_buffers):
        self.unpacked.append((slot_idx, rank, num_blocks, len(layer_buffers)))
        super().unpack_rank_layers(slot_idx, rank, num_blocks, layer_buffers)

    def mlock_async(self, my_rank=None):
        self.mlocked_rank = my_rank

    def close(self):
        self.closed = True
        super().close()


def _make_pool(*, num_slots=2, ranks_per_host=1) -> _RecordingPool:
    pool = _RecordingPool.create(_pool_spec(num_slots, ranks_per_host),
                                 f"test_zmq_shm_{uuid.uuid4().hex}")
    _POOLS.append(pool)
    return pool


def _pool_spec(num_slots=2, ranks_per_host=1) -> PoolSpec:
    return PoolSpec(
        num_slots=num_slots,
        tp_size=ranks_per_host,
        num_layers=_NUM_LAYERS,
        max_blocks=_CACHE_BLOCKS,
        layer_shard_shape=(_CACHE_BLOCKS, ) + _BLOCK_SHAPE,
        dtype=_DTYPE,
    )


def _layer_payload(num_blocks: int, fill: int) -> bytes:
    """Construct a serialized single-layer KV shard matching unpack_rank_layers requirements."""
    elems = num_blocks * _BLOCK_SHAPE[0] * _BLOCK_SHAPE[1]
    return bytes([fill]) * (elems * _ELEM_BYTES)


class _FakeFuture:
    """Asynchronous transport future test double supporting configurable wait errors."""

    def __init__(self, block: threading.Event | None = None, error=None):
        self.block = block
        self.error = error
        self.waited = False

    def wait(self):
        self.waited = True
        if self.block is not None:
            self.block.wait(5.0)
        if self.error is not None:
            raise self.error


class _FakeConnector(zsb.ZmqShmKvConnectorBase):
    """Concrete ZmqShmKvConnectorBase implementation routing transport hooks to CPU tensors."""

    def __init__(self, vllm_config):
        self.stage_calls: list[tuple] = []
        self.stage_sync_calls: list[tuple] = []
        self.sync_calls: list = []
        self.fast_scatter_calls: list[tuple] = []
        self.stage_block: threading.Event | None = None
        self.stage_error: BaseException | None = None
        self.stage_raises: BaseException | None = None
        self.scatter_enable_calls = 0
        super().__init__(vllm_config)

    def _stage_d2h(self, slot_idx, num_blocks, block_ids):
        if self.stage_raises is not None:
            raise self.stage_raises
        self.stage_calls.append((slot_idx, num_blocks, list(block_ids)))
        future = _FakeFuture(block=self.stage_block, error=self.stage_error)
        return future, [torch.zeros(2)], [torch.zeros(2)], 4096

    def _stage_d2h_sync(self, slot_idx, num_blocks, block_ids):
        self.stage_sync_calls.append((slot_idx, num_blocks, list(block_ids)))

    def _wait_stage(self, future):
        future.wait()

    def _h2d_into_device(self, src_views):
        return [v.clone() for v in src_views]

    def _h2d_into_device_async(self, src_views):
        return _FakeFuture(), [v.clone() for v in src_views]

    def _synchronize_device(self, tensor):
        self.sync_calls.append(tensor)

    def _try_fast_scatter(self, device_shards, kv_caches, local_blocks):
        self.fast_scatter_calls.append(
            (len(device_shards), len(kv_caches), list(local_blocks)))
        return [c.clone() for c in kv_caches]

    def _maybe_enable_kv_scatter(self):
        self.scatter_enable_calls += 1


class _ModuleStub:
    """Module wrapper that overrides specified attributes while delegating remaining lookups.

    Patching `zsb.time` or `zsb.os` with this stub scopes overrides strictly to
    the module under test, preventing concurrent background threads from
    leaking scripted clocks or mocked filesystem state across tests.
    """

    def __init__(self, real, **overrides):
        self._real = real
        self._overrides = overrides

    def __getattr__(self, name):
        try:
            return self._overrides[name]
        except KeyError:
            return getattr(self._real, name)


class _InlineExecutor:
    """Synchronous executor that executes submitted callables immediately on the calling thread."""

    def __init__(self):
        self.submitted: list = []

    def submit(self, fn, *args, **kwargs):
        self.submitted.append(fn)
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=False):
        pass


def _frame(data: bytes):
    """Lightweight zmq.Frame stand-in providing a `.buffer` attribute for message deserialization."""
    return SimpleNamespace(buffer=memoryview(data))


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

_LIVE: list = []
_POOLS: list = []


@pytest.fixture(autouse=True)
def _shutdown_connectors(monkeypatch):
    # Clear external cluster topology environment variables to ensure
    # rank-to-host arithmetic evaluates with deterministic single-host defaults.
    monkeypatch.delenv("TPU_NUM_HOSTS", raising=False)
    yield
    while _LIVE:
        conn = _LIVE.pop()
        conn._stop_event.set()
        try:
            conn._coord_teardown()
        except Exception:
            pass
    while _POOLS:
        # Unlink shared memory segments even if tests fail mid-execution to
        # prevent resource leaks in /dev/shm.
        try:
            _POOLS.pop().close()
        except Exception:
            pass


def _make_vllm_config(*,
                      is_producer=True,
                      block_size=1,
                      max_model_len=_CACHE_BLOCKS,
                      dp_rank=0):
    cfg = MagicMock()
    cfg.kv_transfer_config.is_kv_producer = is_producer
    cfg.cache_config.block_size = block_size
    cfg.model_config.max_model_len = max_model_len
    cfg.parallel_config.data_parallel_rank = dp_rank
    cfg.compilation_config.static_forward_context = {}
    return cfg


def _make_conn(cls=_FakeConnector,
               *,
               tp_rank=0,
               tp_size=1,
               is_producer=True,
               dp_rank=0,
               node_id=0,
               channel_number=0,
               latency_interval=0.0,
               coord_workers=0,
               channel_workers=0,
               stage_pool_size=0,
               stage_wait_timeout=30.0,
               cfg=None):
    cfg = cfg if cfg is not None else _make_vllm_config(
        is_producer=is_producer, dp_rank=dp_rank)
    ctx = _FakeContext()
    with patch(f"{_BASE}.get_tensor_model_parallel_rank", return_value=tp_rank), \
         patch(f"{_BASE}.get_tensor_model_parallel_world_size", return_value=tp_size), \
         patch(f"{_BASE}.dist_utils.get_node_id", return_value=node_id), \
         patch(f"{_BASE}.dist_utils.get_host_ip", return_value="10.0.0.1"), \
         patch(f"{_BASE}.dist_utils.get_kv_transfer_port", return_value="9100"), \
         patch(f"{_BASE}.dist_utils.get_side_channel_port", return_value="9600"), \
         patch(f"{_BASE}.dist_utils.get_transfer_channel_number",
               return_value=channel_number), \
         patch(f"{_BASE}.dist_utils.get_kv_latency_log_interval",
               return_value=latency_interval), \
         patch(f"{_BASE}.dist_utils.get_kv_coord_executor_max_workers",
               return_value=coord_workers), \
         patch(f"{_BASE}.dist_utils.get_kv_channel_executor_max_workers",
               return_value=channel_workers), \
         patch(f"{_BASE}.dist_utils.get_kv_stage_waiter_pool_size",
               return_value=stage_pool_size), \
         patch(f"{_BASE}.dist_utils.get_kv_stage_wait_timeout_secs",
               return_value=stage_wait_timeout), \
         patch(f"{_BASE}.zmq",
               new=_ModuleStub(zmq, Context=lambda *a, **kw: ctx)):
        conn = cls(cfg)
    _LIVE.append(conn)
    return conn


def _make_runner(num_layers=_NUM_LAYERS):
    return SimpleNamespace(
        device=torch.device("cpu"),
        kv_caches=[
            torch.zeros((_CACHE_BLOCKS, ) + _BLOCK_SHAPE, dtype=torch.float32)
            for _ in range(num_layers)
        ],
    )


def _attach_runner(conn,
                   *,
                   ranks_per_host=1,
                   num_slots=2,
                   runner=None,
                   pool=True):
    """Initialize connector runner state normally configured during register_runner."""
    conn.runner = runner if runner is not None else _make_runner()
    conn.device = conn.runner.device
    conn._extract_kv_layout()
    if pool:
        conn._coord_pool = _make_pool(num_slots=num_slots,
                                      ranks_per_host=ranks_per_host)
        conn._coord_pool_spec = conn._coord_pool.spec
    return conn


def _drain_ipc(conn) -> list[tuple]:
    """Drain all messages currently queued for the IPC thread as (ident, tag, payload) tuples."""
    out = []
    while True:
        try:
            ident, tag, payload = conn._coord_ipc_out.get_nowait()
        except queue.Empty:
            return out
        out.append((ident, tag, zsb._secure_loads(payload)))


def _send_entry(conn, uuid=7, req_id="r0", slot_idx=0, num_blocks=2, ttl=60.0):
    entry = zsb._CoordSendEntry(req_id=req_id,
                                slot_idx=slot_idx,
                                num_blocks=num_blocks,
                                expiration_time=time.perf_counter() + ttl,
                                tp_size=conn.tp_size)
    conn._coord_send[uuid] = entry
    return entry


def _recv_entry(conn,
                uuid=11,
                req_id="d0",
                slot_idx=0,
                num_blocks=2,
                remote_host="10.0.0.2",
                remote_port=9100,
                side_port=None):
    entry = zsb._CoordRecvEntry(req_id=req_id,
                                uuid=uuid,
                                slot_idx=slot_idx,
                                num_blocks=num_blocks,
                                local_blocks=[0, 1],
                                remote_blocks=[4, 5],
                                remote_host=remote_host,
                                remote_port=remote_port,
                                remote_side_channel_port=side_port)
    conn._coord_recv[uuid] = entry
    return entry


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _derived_key(machine_id: str) -> bytes:
    secret = f"vllm-torchtpu-ipc-auth-{machine_id}-{os.getuid()}"
    return hashlib.sha256(secret.encode("utf-8")).digest()


class TestIpcAuth:

    def test_env_key_wins(self, monkeypatch):
        monkeypatch.setenv("VLLM_TORCHTPU_IPC_KEY", "hunter2")
        assert zsb._get_default_ipc_key() == b"hunter2"

    def test_derived_key_reads_machine_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("VLLM_TORCHTPU_IPC_KEY", raising=False)
        machine_id = tmp_path / "machine-id"
        machine_id.write_text("abc123\n")
        fake_path = _ModuleStub(os.path,
                                exists=lambda path: path == "/etc/machine-id")
        with patch(f"{_BASE}.os", new=_ModuleStub(os, path=fake_path)), \
             patch(f"{_BASE}.open", create=True,
                   return_value=open(machine_id, "r")):
            key = zsb._get_default_ipc_key()
        assert key == _derived_key("abc123")

    def test_derived_key_without_machine_id(self, monkeypatch):
        monkeypatch.delenv("VLLM_TORCHTPU_IPC_KEY", raising=False)
        fake_path = _ModuleStub(os.path, exists=lambda _path: False)
        with patch(f"{_BASE}.os", new=_ModuleStub(os, path=fake_path)):
            key = zsb._get_default_ipc_key()
        assert key == _derived_key("")

    def test_derived_key_survives_unreadable_machine_id(self, monkeypatch):
        monkeypatch.delenv("VLLM_TORCHTPU_IPC_KEY", raising=False)
        fake_path = _ModuleStub(os.path, exists=lambda _path: True)
        with patch(f"{_BASE}.os", new=_ModuleStub(os, path=fake_path)), \
             patch(f"{_BASE}.open", create=True, side_effect=OSError("nope")):
            key = zsb._get_default_ipc_key()
        assert key == _derived_key("")

    def test_secure_roundtrip(self):
        payload = {"uuid": 5, "blocks": [1, 2, 3]}
        assert zsb._secure_loads(zsb._secure_dumps(payload)) == payload

    def test_secure_loads_rejects_short_payload(self):
        with pytest.raises(ValueError, match="too short"):
            zsb._secure_loads(b"\x00" * 8)

    def test_secure_loads_rejects_tampered_payload(self):
        blob = bytearray(zsb._secure_dumps({"a": 1}))
        blob[-1] ^= 0xFF
        with pytest.raises(ValueError, match="HMAC authentication failed"):
            zsb._secure_loads(bytes(blob))


class TestRemoveIpcEndpoint:

    def test_ignores_non_ipc_path(self, tmp_path):
        target = tmp_path / "keep.sock"
        target.write_text("")
        zsb._try_remove_ipc_endpoint(f"tcp://127.0.0.1:{5555}")
        assert target.exists()

    def test_unlinks_stale_socket(self, tmp_path):
        target = tmp_path / "stale.sock"
        target.write_text("")
        zsb._try_remove_ipc_endpoint(f"ipc://{target}")
        assert not target.exists()

    def test_missing_socket_is_not_an_error(self, tmp_path):
        zsb._try_remove_ipc_endpoint(f"ipc://{tmp_path / 'absent.sock'}")

    def test_oserror_is_logged_not_raised(self, tmp_path):
        # Verify non-FileNotFoundError OS exceptions (e.g. EISDIR when unlinking
        # a directory) are handled gracefully with a warning rather than raised.
        with patch(f"{_BASE}.logger") as log:
            zsb._try_remove_ipc_endpoint(f"ipc://{tmp_path}")
        assert tmp_path.is_dir()
        assert log.warning.called


class TestLatencyTracker:

    def test_summary_reports_count_average_min_and_max_per_phase(self):
        tracker = zsb._LatencyTracker("who", log_interval_s=0.01)
        for value in (5.0, 1.0, 9.0):
            tracker.record("d2h", value)
        tracker._last_log = time.perf_counter() - 10.0
        with patch(f"{_BASE}.logger") as log:
            tracker.record("wire", 4.0)
        fmt, who, summary = log.info.call_args.args
        assert "latency summary" in fmt and who == "who"
        assert "d2h: n=3 avg=5.0ms min=1.0ms max=9.0ms" in summary
        assert "wire: n=1 avg=4.0ms min=4.0ms max=4.0ms" in summary

    def test_negative_interval_is_clamped_to_zero(self):
        assert zsb._LatencyTracker("who", log_interval_s=-3.0)._log_interval_s \
            == 0.0

    def test_summary_is_suppressed_inside_the_interval(self):
        tracker = zsb._LatencyTracker("who", log_interval_s=1000.0)
        with patch(f"{_BASE}.logger") as log:
            tracker.record("wire", 4.0)
        assert not log.info.called


class TestCoordEntries:

    def test_send_entry_sizes_one_event_per_rank(self):
        entry = zsb._CoordSendEntry(req_id="r",
                                    slot_idx=0,
                                    num_blocks=1,
                                    expiration_time=0.0,
                                    tp_size=4)
        assert len(entry.staged_events) == 4
        assert not entry.stage_complete.is_set()

    def test_send_entry_keeps_supplied_events(self):
        events = [threading.Event()]
        entry = zsb._CoordSendEntry(req_id="r",
                                    slot_idx=0,
                                    num_blocks=1,
                                    expiration_time=0.0,
                                    tp_size=4,
                                    staged_events=events)
        assert entry.staged_events is events

    def test_recv_entry_defaults(self):
        entry = zsb._CoordRecvEntry(req_id="r",
                                    uuid=1,
                                    slot_idx=0,
                                    num_blocks=1,
                                    local_blocks=[0],
                                    remote_blocks=[1],
                                    remote_host="h",
                                    remote_port=1)
        assert not entry.pull_ok and not entry.reported_done
        assert entry.copied == set()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:

    def test_producer_defaults(self):
        conn = _make_conn()
        assert conn.is_producer
        assert conn.tp_rank == 0 and conn.tp_size == 1
        assert conn.ranks_per_host == 1
        assert conn._is_host_coordinator
        # Single-rank topologies bypass HELLO handshakes; worker readiness is
        # marked immediately.
        assert conn._coord_workers_ready.is_set()
        assert conn._n_channels == 1
        assert conn._kv_transfer_ports == [9100]

    def test_multi_host_split_of_ranks(self, monkeypatch):
        monkeypatch.setenv("TPU_NUM_HOSTS", "2")
        conn = _make_conn(tp_rank=5, tp_size=8)
        assert conn.ranks_per_host == 4
        assert conn.local_tp_rank == 1
        assert not conn._is_host_coordinator
        assert not conn._coord_workers_ready.is_set()

    def test_local_rank_zero_on_second_host_is_a_coordinator(
            self, monkeypatch):
        monkeypatch.setenv("TPU_NUM_HOSTS", "2")
        conn = _make_conn(tp_rank=4, tp_size=8)
        assert conn.local_tp_rank == 0
        assert conn._is_host_coordinator

    def test_explicit_executor_sizes_are_honoured(self):
        conn = _make_conn(tp_size=4,
                          coord_workers=3,
                          channel_workers=5,
                          stage_pool_size=2)
        assert conn._coord_executor._max_workers == 3
        assert conn._coord_channel_executor._max_workers == 5
        assert conn._coord_stage_waiter_pool._max_workers == 2

    def test_derived_executor_sizes(self):
        conn = _make_conn(tp_size=4)
        assert conn._coord_executor._max_workers == 2
        assert conn._coord_channel_executor._max_workers == 8
        assert conn._coord_stage_waiter_pool._max_workers == 4

    def test_local_to_global_rank_seeded_with_self(self):
        conn = _make_conn(tp_rank=3, tp_size=4)
        assert conn._coord_local_to_global_rank == {3: 3}

    def test_del_stops_threads_and_destroys_context(self):
        conn = _make_conn()
        _LIVE.remove(conn)
        ctx = conn.zmq_cxt
        conn.__del__()
        assert conn._stop_event.is_set()
        assert ctx.destroyed

    def test_del_before_the_context_exists(self):
        conn = _make_conn()
        _LIVE.remove(conn)
        del conn.zmq_cxt
        conn.__del__()
        assert conn._stop_event.is_set()


# ---------------------------------------------------------------------------
# Bring-up
# ---------------------------------------------------------------------------


class TestRegisterRunner:

    def test_extracts_layout_and_runs_setup(self):
        conn = _make_conn()
        runner = _make_runner()
        with patch.object(conn, "_coord_setup") as setup, \
             patch(f"{_BASE}.dist_utils.get_kv_warmup_enabled",
                   return_value=False):
            conn.register_runner(runner)
        assert conn.num_layers == _NUM_LAYERS
        assert conn.shape == [_CACHE_BLOCKS, *_BLOCK_SHAPE]
        assert conn.dtype == torch.float32
        assert conn.device == runner.device
        assert setup.called
        assert conn.scatter_enable_calls == 1

    def test_rejects_a_runner_without_kv_caches(self):
        conn = _make_conn()
        runner = SimpleNamespace(device=torch.device("cpu"), kv_caches=[])
        with pytest.raises(AssertionError, match="kv_caches"):
            conn.register_runner(runner)

    def test_warmup_runs_when_enabled(self):
        conn = _make_conn()
        with patch.object(conn, "_coord_setup"), \
             patch.object(conn, "_warmup_kv_ops") as warmup, \
             patch(f"{_BASE}.dist_utils.get_kv_warmup_enabled",
                   return_value=True):
            conn.register_runner(_make_runner())
        assert warmup.called

    def test_warmup_failure_is_swallowed(self):
        conn = _make_conn()
        with patch.object(conn, "_coord_setup"), \
             patch.object(conn, "_warmup_kv_ops",
                          side_effect=RuntimeError("compile blew up")), \
             patch(f"{_BASE}.dist_utils.get_kv_warmup_enabled",
                   return_value=True):
            conn.register_runner(_make_runner())


class TestWarmup:

    def test_block_sizes_round_up_to_whole_blocks(self):
        cfg = _make_vllm_config(block_size=16, max_model_len=33)
        conn = _make_conn(cfg=cfg)
        assert conn._warmup_block_sizes() == [3]

    def test_producer_warmup_uses_the_sync_stage_path(self):
        conn = _attach_runner(_make_conn())
        conn._warmup_kv_ops()
        assert conn.stage_sync_calls == [(0, _CACHE_BLOCKS,
                                          list(range(_CACHE_BLOCKS)))]

    def test_zero_block_sizes_are_skipped(self):
        conn = _attach_runner(_make_conn())
        with patch.object(conn, "_warmup_block_sizes", return_value=[0]):
            conn._warmup_kv_ops()
        assert conn.stage_sync_calls == []

    def test_consumer_warmup_naive_path(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._warmup_coord_once(2)
        assert conn.fast_scatter_calls == []
        assert len(conn.sync_calls) == _NUM_LAYERS

    def test_consumer_warmup_fast_scatter_path(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._kv_scatter_enabled = True
        originals = list(conn.runner.kv_caches)
        conn._warmup_coord_once(2)
        assert conn.fast_scatter_calls == [(_NUM_LAYERS, _NUM_LAYERS, [0, 1])]
        assert all(new is not old
                   for new, old in zip(conn.runner.kv_caches, originals))


class TestPoolSpec:

    def test_spec_is_derived_from_the_budget(self):
        cfg = _make_vllm_config(block_size=1, max_model_len=_CACHE_BLOCKS)
        conn = _attach_runner(_make_conn(cfg=cfg))
        with patch(f"{_BASE}.dist_utils.get_kv_shm_pool_gb", return_value=1.0):
            spec = conn._build_pool_spec()
        assert spec.num_layers == _NUM_LAYERS
        assert spec.max_blocks == _CACHE_BLOCKS
        assert spec.layer_shard_shape == (_CACHE_BLOCKS, ) + _BLOCK_SHAPE
        assert spec.dtype == torch.float32
        assert spec.num_slots >= 1

    def test_num_slots_never_drops_below_one(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.dist_utils.get_kv_shm_pool_gb", return_value=0.0):
            assert conn._build_pool_spec().num_slots == 1

    def test_pool_create_and_attach_delegate_to_host_kv_shm_pool(self):
        conn = _make_conn()
        spec = object()
        with patch(f"{_BASE}.HostKVShmPool") as pool_cls:
            assert conn._pool_create(spec, "n") is pool_cls.create.return_value
            assert conn._pool_attach(spec, "n") is pool_cls.attach.return_value
        pool_cls.create.assert_called_once_with(spec, "n")
        pool_cls.attach.assert_called_once_with(spec, "n")


class TestCoordSetup:

    def _setup(self, conn, *, pin=False):
        """Execute _coord_setup with termination events preset to suppress background threads."""
        conn._stop_event.set()
        with patch(f"{_BASE}.dist_utils.get_kv_shm_pool_gb", return_value=1.0), \
             patch(f"{_BASE}.dist_utils.get_kv_pin_shm", return_value=pin), \
             patch(f"{_BASE}.dist_utils.get_shm_name", return_value="shm0"), \
             patch(f"{_BASE}.dist_utils.get_ipc_socket_path",
                   return_value="ipc:///tmp/tpu-test.sock"), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1), \
             patch(f"{_BASE}.make_zmq_socket", return_value=_FakeSocket()), \
             patch(f"{_BASE}._try_remove_ipc_endpoint"), \
             patch.object(conn, "_pool_create",
                          side_effect=lambda spec, name: _make_pool()), \
             patch.object(conn, "_pool_attach",
                          side_effect=lambda spec, name: _make_pool()):
            conn._coord_setup()

    def test_rank0_producer_binds_ipc_and_spawns_loops(self):
        conn = _attach_runner(_make_conn(), pool=False)
        self._setup(conn)
        assert conn._coord_ipc_sock.bound == "ipc:///tmp/tpu-test.sock"
        assert conn._coord_ipc_sock.socket_type == zmq.ROUTER
        names = {t.name for t in conn._coord_threads}
        assert {
            "tpu_conn_ipc", "tpu_conn_stage_wait", "tpu_conn_expire",
            "tpu_conn_data_ch0", "tpu_conn_notif"
        } <= names

    def test_rank0_consumer_has_no_external_servers(self):
        conn = _attach_runner(_make_conn(is_producer=False), pool=False)
        self._setup(conn)
        names = {t.name for t in conn._coord_threads}
        assert "tpu_conn_data_ch0" not in names
        assert conn._coord_notif_sockets == {}

    def test_pin_shm_locks_only_this_rank_slice(self):
        conn = _attach_runner(_make_conn(), pool=False)
        self._setup(conn, pin=True)
        assert conn._coord_pool.mlocked_rank == 0

    def test_non_coordinator_attaches_and_says_hello(self):
        conn = _attach_runner(_make_conn(tp_rank=1, tp_size=2), pool=False)
        self._setup(conn)
        assert conn._coord_ipc_sock.connected == "ipc:///tmp/tpu-test.sock"
        assert conn._coord_ipc_sock.opts[zmq.IDENTITY] == b"tpu_rank_1"
        assert _drain_ipc(conn) == [(None, zsb._IPC_HELLO, (1, 1))]

    def test_attach_retries_until_the_pool_appears(self):
        conn = _attach_runner(_make_conn(tp_rank=1, tp_size=2))
        pool = _make_pool()
        attempts = {"n": 0}

        def flaky(spec, name):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise FileNotFoundError(name)
            return pool

        with patch.object(conn, "_pool_attach", side_effect=flaky), \
             patch(f"{_BASE}.time", new=_ModuleStub(time, sleep=lambda _s: None)):
            assert conn._coord_attach_shm_with_retry(object(), "shm") is pool
        assert attempts["n"] == 3

    def test_attach_gives_up_after_the_deadline(self):
        conn = _make_conn(tp_rank=1, tp_size=2)
        # Simulate two retries within the deadline before advancing time past
        # the timeout threshold.
        clock = itertools.chain([0.0, 0.0], itertools.repeat(99.0))
        with patch.object(conn, "_pool_attach",
                          side_effect=FileNotFoundError("shm")), \
             patch(f"{_BASE}.time",
                   new=_ModuleStub(time,
                                   sleep=lambda _s: None,
                                   perf_counter=lambda: next(clock))), \
             pytest.raises(RuntimeError, match="could not attach"):
            conn._coord_attach_shm_with_retry(object(), "shm")


# ---------------------------------------------------------------------------
# IPC envelope plumbing
# ---------------------------------------------------------------------------


class TestIpcEnvelopes:

    def test_worker_send_enqueues_an_unaddressed_frame(self):
        conn = _make_conn(tp_rank=1, tp_size=2)
        conn._coord_send_ipc(zsb._IPC_COPY_DONE, (9, 1))
        assert _drain_ipc(conn) == [(None, zsb._IPC_COPY_DONE, (9, 1))]

    def test_broadcast_addresses_every_known_rank(self):
        conn = _make_conn(tp_size=3)
        conn._coord_rank_to_identity = {1: b"r1", 2: b"r2"}
        conn._coord_broadcast(zsb._IPC_DROP, (4, ))
        assert sorted(_drain_ipc(conn)) == [(b"r1", zsb._IPC_DROP, (4, )),
                                            (b"r2", zsb._IPC_DROP, (4, ))]

    def test_broadcast_with_no_peers_sends_nothing(self):
        conn = _make_conn()
        conn._coord_broadcast(zsb._IPC_DROP, (4, ))
        assert _drain_ipc(conn) == []

    def test_drain_sends_dealer_and_router_shapes(self):
        conn = _make_conn()
        sock = _FakeSocket()
        conn._coord_ipc_sock = sock
        conn._coord_ipc_out.put((None, b"T", b"d1"))
        conn._coord_ipc_out.put((b"ident", b"T", b"d2"))
        conn._coord_ipc_drain_outbound()
        assert sock.sent == [[b"T", b"d1"], [b"ident", b"T", b"d2"]]

    def test_drain_keeps_going_after_a_send_error(self):
        conn = _make_conn()
        conn._coord_ipc_sock = _FakeSocket(send_error=zmq.ZMQError("down"))
        conn._coord_ipc_out.put((None, b"T", b"d1"))
        conn._coord_ipc_drain_outbound()
        assert _drain_ipc(conn) == []


class TestTeardown:

    def test_teardown_is_safe_before_setup(self):
        conn = _make_conn()
        _LIVE.remove(conn)
        conn._coord_teardown()
        assert conn._stop_event.is_set()

    def test_teardown_before_init_finished(self):
        conn = _make_conn()
        _LIVE.remove(conn)
        for attr in ("_stage_pending_q", "_coord_executor",
                     "_coord_channel_executor", "_coord_stage_waiter_pool"):
            delattr(conn, attr)
        conn._coord_teardown()
        assert conn._stop_event.is_set()

    def test_a_full_stage_queue_does_not_block_teardown(self):
        conn = _attach_runner(_make_conn())
        _LIVE.remove(conn)
        conn._stage_pending_q = SimpleNamespace(put_nowait=MagicMock(
            side_effect=queue.Full()))
        conn._coord_teardown()
        assert conn._coord_pool.closed

    def test_teardown_closes_the_pool_and_wakes_the_waiter(self):
        conn = _attach_runner(_make_conn())
        _LIVE.remove(conn)
        pool = conn._coord_pool
        conn._coord_teardown()
        assert pool.closed
        assert conn._stage_pending_q.get_nowait() is None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestProcessSendLoad:

    def test_coordinator_handles_sends_and_loads(self):
        conn = _attach_runner(_make_conn())
        meta = TPUConnectorMetadata(
            reqs_to_send={
                "r1": SendMeta(uuid=1,
                               local_block_ids=[0, 1],
                               expiration_time=0.0)
            },
            reqs_to_load={},
        )
        with patch.object(conn, "_coord_rank0_handle_new_send") as handle:
            conn.process_send_load(meta)
        handle.assert_called_once()

    def test_coordinator_waits_on_hellos_before_broadcasting(self):
        conn = _attach_runner(_make_conn(tp_size=2))
        conn._coord_workers_ready = _ScriptedEvent([False])
        meta = TPUConnectorMetadata(
            reqs_to_send={
                "r1": SendMeta(uuid=1,
                               local_block_ids=[0],
                               expiration_time=0.0)
            },
            reqs_to_load={},
        )
        with patch.object(conn, "_coord_rank0_handle_new_send") as handle:
            conn._coord_process_send_load(meta)
        assert handle.called

    def test_empty_metadata_skips_the_hello_gate(self):
        conn = _attach_runner(_make_conn())
        gate = MagicMock()
        conn._coord_workers_ready = gate
        conn._coord_process_send_load(TPUConnectorMetadata())
        assert not gate.wait.called

    def test_consumer_coordinator_dispatches_loads(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        meta = TPUConnectorMetadata(
            reqs_to_load={
                "d1":
                LoadMeta(uuid=2,
                         local_block_ids=[0],
                         remote_block_ids=[3],
                         remote_host="h",
                         remote_port=1)
            })
        with patch.object(conn, "_coord_rank0_handle_new_load") as handle:
            conn._coord_process_send_load(meta)
        handle.assert_called_once()

    def test_non_coordinator_stages_inline(self):
        conn = _attach_runner(_make_conn(tp_rank=1, tp_size=2))
        meta = TPUConnectorMetadata(reqs_to_send={
            "r1":
            SendMeta(uuid=1, local_block_ids=[0], expiration_time=0.0)
        })
        with patch.object(conn, "_coord_worker_stage_inline") as stage:
            conn._coord_process_send_load(meta)
        stage.assert_called_once_with(1, "r1")

    def test_drain_pass_runs_when_remote_blocks_are_none(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        meta = TPUConnectorMetadata(
            reqs_to_load={
                "d1":
                LoadMeta(uuid=2,
                         local_block_ids=[0],
                         remote_block_ids=None,
                         remote_host="h",
                         remote_port=1)
            })
        with patch.object(conn, "_coord_rank0_handle_new_load"), \
             patch.object(conn, "_coord_drain_scatter") as drain:
            conn._coord_process_send_load(meta)
        drain.assert_called_once()


class TestSmallHelpers:

    def test_blocks_token_counts_blocks(self):
        assert _make_conn()._blocks_token([1, 2, 3]) == 3

    def test_pull_response_header(self):
        conn = _attach_runner(_make_conn(tp_size=1))
        entry = _send_entry(conn, num_blocks=5)
        header = conn._pull_response_header(entry)
        assert header == {
            "tp_size": 1,
            "num_layers": _NUM_LAYERS,
            "num_blocks": 5,
            "dtype": str(torch.float32),
            "layer_shard_shape": list(_BLOCK_SHAPE),
        }

    def test_replace_runner_kv_cache_tolerates_a_non_dict_context(self):
        conn = _attach_runner(_make_conn())
        conn.vllm_config.compilation_config.static_forward_context = []
        new = torch.ones_like(conn.runner.kv_caches[1])
        conn._replace_runner_kv_cache(1, new)
        assert conn.runner.kv_caches[1] is new

    def test_maybe_enable_kv_scatter_is_a_no_op_on_the_base(self):
        conn = _make_conn(cls=zsb.ZmqShmKvConnectorBase)
        assert conn._maybe_enable_kv_scatter() is None
        assert conn._kv_scatter_enabled is False

    def test_get_finished_delegates_to_the_coordinator_path(self):
        conn = _make_conn()
        with patch.object(conn,
                          "_coord_get_finished",
                          return_value=({"a"}, {"b"})) as inner:
            assert conn.get_finished({"x"}) == ({"a"}, {"b"})
        assert inner.called


class TestAbstractHooks:

    @pytest.mark.parametrize(("name", "args"), (
        ("_stage_d2h", (0, 1, [0])),
        ("_stage_d2h_sync", (0, 1, [0])),
        ("_wait_stage", (None, )),
        ("_h2d_into_device", ([], )),
        ("_h2d_into_device_async", ([], )),
        ("_synchronize_device", (None, )),
        ("_try_fast_scatter", ([], [], [])),
    ))
    def test_transport_hooks_are_abstract(self, name, args):
        conn = _make_conn(cls=zsb.ZmqShmKvConnectorBase)
        with pytest.raises(NotImplementedError):
            getattr(conn, name)(*args)


# ---------------------------------------------------------------------------
# Producer: staging
# ---------------------------------------------------------------------------


class TestHandleNewSend:

    def test_acquires_a_slot_broadcasts_and_stages(self):
        conn = _attach_runner(_make_conn(tp_size=2))
        conn._coord_rank_to_identity = {1: b"r1"}
        meta = SendMeta(uuid=42,
                        local_block_ids=[0, 1, 2],
                        expiration_time=time.perf_counter() + 30)
        conn._coord_rank0_handle_new_send("req-a", meta)
        entry = conn._coord_send[42]
        assert (entry.req_id, entry.slot_idx, entry.num_blocks) == ("req-a", 0,
                                                                    3)
        assert len(entry.staged_events) == 2
        assert _drain_ipc(conn) == [(b"r1", zsb._IPC_STAGE_NOTIFY,
                                     (42, 0, 3, [0, 1, 2]))]
        assert conn.stage_calls == [(0, 3, [0, 1, 2])]

    def test_pool_exhaustion_surfaces_done_sending(self):
        conn = _attach_runner(_make_conn(), num_slots=1)
        conn._coord_pool.acquire_slot()
        _send_entry(conn, uuid=1, req_id="older")
        _recv_entry(conn, uuid=2)
        meta = SendMeta(uuid=42, local_block_ids=[0], expiration_time=0.0)
        conn._coord_rank0_handle_new_send("req-a", meta)
        assert conn._coord_done_sending == {"req-a"}
        assert 42 not in conn._coord_send
        assert conn.stage_calls == []

    def test_stage_failure_is_logged_and_leaves_the_entry_unsignalled(self):
        # TODO(#771): Rank 0 currently only logs exceptions when stage_shard
        # raises without broadcasting STAGE_DONE, causing inbound PULL requests
        # for this UUID to block until timing out with ERR. Unlike worker ranks
        # in _coord_worker_stage_inline which signal failed=True, rank 0 does not
        # propagate failure. This test pins existing behavior until harmonized.
        conn = _attach_runner(_make_conn())
        conn.stage_raises = RuntimeError("dma refused")
        meta = SendMeta(uuid=42, local_block_ids=[0], expiration_time=0.0)
        conn._coord_rank0_handle_new_send("req-a", meta)
        entry = conn._coord_send[42]
        assert not entry.stage_failed
        assert not entry.staged_events[0].is_set()

    def test_non_producer_asserts(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        meta = SendMeta(uuid=1, local_block_ids=[0], expiration_time=0.0)
        with pytest.raises(AssertionError):
            conn._coord_rank0_handle_new_send("r", meta)


class TestWorkerStageInline:

    def test_stages_once_stage_notify_lands(self):
        conn = _attach_runner(_make_conn(tp_rank=1, tp_size=2))
        conn._worker_pending_stage[5] = (1, 3, [7, 8, 9])
        conn._coord_worker_stage_inline(5, "req-a", timeout=1.0)
        assert conn.stage_calls == [(1, 3, [7, 8, 9])]
        assert 5 not in conn._worker_pending_stage

    def test_timeout_signals_a_failed_stage(self):
        conn = _attach_runner(_make_conn(tp_rank=1, tp_size=2))
        conn._coord_worker_stage_inline(5, "req-a", timeout=0.01)
        assert _drain_ipc(conn) == [(None, zsb._IPC_STAGE_DONE, (5, 1, True))]

    def test_stage_exception_signals_a_failed_stage(self):
        conn = _attach_runner(_make_conn(tp_rank=1, tp_size=2))
        conn.stage_raises = RuntimeError("boom")
        conn._worker_pending_stage[5] = (0, 1, [0])
        conn._coord_worker_stage_inline(5, "req-a", timeout=1.0)
        assert _drain_ipc(conn) == [(None, zsb._IPC_STAGE_DONE, (5, 1, True))]

    def test_a_late_stage_notify_still_wins(self):
        conn = _attach_runner(_make_conn(tp_rank=1, tp_size=2))

        def deliver():
            time.sleep(0.05)
            with conn._worker_pending_stage_cv:
                conn._worker_pending_stage[5] = (0, 1, [0])
                conn._worker_pending_stage_cv.notify_all()

        thread = threading.Thread(target=deliver)
        thread.start()
        try:
            conn._coord_worker_stage_inline(5, "req-a", timeout=5.0)
        finally:
            thread.join()
        assert conn.stage_calls == [(0, 1, [0])]


class TestStageShardAndWaiter:

    def test_stage_shard_enqueues_an_in_flight_entry(self):
        conn = _attach_runner(_make_conn())
        conn._coord_stage_shard(9, 1, 2, [0, 1])
        entry = conn._stage_pending_q.get_nowait()
        assert (entry.uuid, entry.slot_idx, entry.num_blocks) == (9, 1, 2)
        assert entry.total_bytes == 4096
        assert entry.tpu_tensors and entry.cpu_tensors

    def test_wait_worker_records_latency_and_signals(self):
        conn = _attach_runner(_make_conn())
        send = _send_entry(conn, uuid=9)
        conn._coord_stage_shard(9, 0, 2, [0, 1])
        conn._coord_stage_wait_worker(conn._stage_pending_q.get_nowait())
        assert send.stage_complete.is_set()
        assert not send.stage_failed
        assert conn._lat._stats[zsb._LAT_D2H][0] == 1.0

    def test_wait_worker_marks_a_raising_future_failed(self):
        conn = _attach_runner(_make_conn())
        send = _send_entry(conn, uuid=9)
        conn.stage_error = RuntimeError("dma fault")
        conn._coord_stage_shard(9, 0, 2, [0, 1])
        conn._coord_stage_wait_worker(conn._stage_pending_q.get_nowait())
        assert send.stage_failed

    def test_complete_stage_entry_runs_once(self):
        conn = _attach_runner(_make_conn())
        send = _send_entry(conn, uuid=9)
        conn._coord_stage_shard(9, 0, 2, [0, 1])
        entry = conn._stage_pending_q.get_nowait()
        conn._complete_stage_entry(entry, failed=False)
        assert entry.tpu_tensors == [] and entry.cpu_tensors == []
        send.stage_failed = False
        conn._complete_stage_entry(entry, failed=True)
        assert not send.stage_failed

    def test_waiter_loop_returns_on_the_shutdown_sentinel(self):
        conn = _attach_runner(_make_conn())
        conn._stage_pending_q.put(None)
        conn._coord_stage_waiter_loop()

    def test_waiter_loop_dispatches_entries_to_the_pool(self):
        conn = _attach_runner(_make_conn())
        send = _send_entry(conn, uuid=9)
        conn._coord_stage_shard(9, 0, 2, [0, 1])
        thread = threading.Thread(target=conn._coord_stage_waiter_loop)
        thread.start()
        try:
            assert send.stage_complete.wait(5.0)
        finally:
            conn._stop_event.set()
            thread.join(5.0)
        assert not thread.is_alive()
        assert not send.stage_failed

    def test_waiter_loop_keeps_an_entry_inside_its_deadline(self):
        conn = _attach_runner(_make_conn())
        send = _send_entry(conn, uuid=9)
        blocker = threading.Event()
        conn.stage_block = blocker
        conn._coord_stage_wait_timeout_s = 30.0
        conn._coord_stage_shard(9, 0, 2, [0, 1])
        # Enqueue a None sentinel behind the work item to terminate the stage
        # waiter loop after one sweep.
        conn._stage_pending_q.put(None)
        try:
            conn._coord_stage_waiter_loop()
            assert not send.stage_failed
            assert not send.stage_complete.is_set()
        finally:
            blocker.set()
        assert send.stage_complete.wait(5.0)
        assert not send.stage_failed

    def test_waiter_loop_fails_an_entry_past_its_deadline(self):
        conn = _attach_runner(_make_conn())
        send = _send_entry(conn, uuid=9)
        blocker = threading.Event()
        conn.stage_block = blocker
        conn._coord_stage_wait_timeout_s = 0.0
        conn._coord_stage_shard(9, 0, 2, [0, 1])
        original = conn._complete_stage_entry

        def stop_after_complete(entry, failed):
            original(entry, failed)
            conn._stop_event.set()

        conn._complete_stage_entry = stop_after_complete
        try:
            conn._coord_stage_waiter_loop()
        finally:
            blocker.set()
        assert send.stage_failed


class TestSignalStageDone:

    def test_coordinator_marks_its_own_rank(self):
        conn = _attach_runner(_make_conn())
        entry = _send_entry(conn, uuid=3)
        conn._signal_stage_done(3)
        assert entry.staged == {0}
        assert entry.stage_complete.is_set()
        assert entry.staged_events[0].is_set()

    def test_coordinator_records_the_failure_flag(self):
        conn = _attach_runner(_make_conn())
        entry = _send_entry(conn, uuid=3)
        conn._signal_stage_done(3, failed=True)
        assert entry.stage_failed

    def test_unknown_uuid_is_ignored(self):
        conn = _attach_runner(_make_conn())
        conn._signal_stage_done(999)

    def test_an_entry_without_per_rank_events_still_completes(self):
        conn = _attach_runner(_make_conn())
        entry = zsb._CoordSendEntry(req_id="r",
                                    slot_idx=0,
                                    num_blocks=1,
                                    expiration_time=0.0,
                                    tp_size=1)
        entry.staged_events = []
        conn._coord_send[3] = entry
        conn._signal_stage_done(3)
        assert entry.staged == {0}
        assert entry.stage_complete.is_set()

    def test_partial_staging_leaves_stage_complete_clear(self):
        conn = _attach_runner(_make_conn(tp_size=2))
        entry = _send_entry(conn, uuid=3)
        conn._signal_stage_done(3)
        assert entry.staged == {0}
        assert not entry.stage_complete.is_set()

    def test_non_coordinator_sends_over_ipc(self):
        conn = _attach_runner(_make_conn(tp_rank=1, tp_size=2))
        conn._signal_stage_done(3, failed=True)
        assert _drain_ipc(conn) == [(None, zsb._IPC_STAGE_DONE, (3, 1, True))]


# ---------------------------------------------------------------------------
# Consumer: admission and pull
# ---------------------------------------------------------------------------


def _load_meta(uuid=11,
               local_blocks=(0, 1),
               remote_blocks=(4, 5),
               host="10.0.0.2",
               port=9100,
               side_port=None):
    return LoadMeta(uuid=uuid,
                    local_block_ids=list(local_blocks),
                    remote_block_ids=None
                    if remote_blocks is None else list(remote_blocks),
                    remote_host=host,
                    remote_port=port,
                    remote_side_channel_port=side_port)


class TestHandleNewLoad:

    def test_admits_a_new_load_and_submits_the_pull(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_executor = _InlineExecutor()
        with patch.object(conn, "_coord_rank0_pull") as pull:
            conn._coord_rank0_handle_new_load("d1", _load_meta())
        entry = conn._coord_recv[11]
        assert entry.slot_idx == 0 and entry.num_blocks == 2
        assert entry.remote_host == "10.0.0.2" and entry.remote_port == 9100
        pull.assert_called_once_with(entry)

    def test_full_cache_hit_broadcasts_load_skip(self):
        conn = _attach_runner(_make_conn(is_producer=False, tp_size=2))
        conn._coord_rank_to_identity = {1: b"r1"}
        conn._coord_rank0_handle_new_load("d1", _load_meta(remote_blocks=None))
        assert _drain_ipc(conn) == [(b"r1", zsb._IPC_LOAD_SKIP, (11, ))]

    def test_drain_tick_for_a_known_uuid_broadcasts_nothing(self):
        conn = _attach_runner(_make_conn(is_producer=False, tp_size=2))
        conn._coord_rank_to_identity = {1: b"r1"}
        _recv_entry(conn, uuid=11)
        conn._coord_rank0_handle_new_load("d1", _load_meta(remote_blocks=None))
        assert _drain_ipc(conn) == []

    def test_preempted_reemit_updates_blocks_while_the_pull_is_in_flight(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        entry = _recv_entry(conn, uuid=11)
        conn._coord_rank0_handle_new_load("d1-again",
                                          _load_meta(local_blocks=(6, 7, 8)))
        assert entry.local_blocks == [6, 7, 8]
        assert entry.num_blocks == 3
        assert entry.req_id == "d1-again"
        assert _drain_ipc(conn) == []

    def test_preempted_reemit_after_the_pull_rebroadcasts_load_notify(self):
        conn = _attach_runner(_make_conn(is_producer=False, tp_size=2))
        conn._coord_rank_to_identity = {1: b"r1"}
        entry = _recv_entry(conn, uuid=11)
        entry.load_complete.set()
        conn._coord_rank0_handle_new_load("d1-again",
                                          _load_meta(local_blocks=(6, 7)))
        [(_ident, tag, notify)] = _drain_ipc(conn)
        assert (tag, notify) == (zsb._IPC_LOAD_NOTIFY, (11, 0, 2, [6, 7]))

    def test_pool_exhaustion_surfaces_done_recving(self):
        conn = _attach_runner(_make_conn(is_producer=False), num_slots=1)
        conn._coord_pool.acquire_slot()
        _recv_entry(conn, uuid=1)
        _send_entry(conn, uuid=2)
        conn._coord_rank0_handle_new_load("d1", _load_meta())
        assert conn._coord_done_recving == {"d1"}
        assert 11 not in conn._coord_recv

    def test_producer_asserts(self):
        conn = _attach_runner(_make_conn())
        with pytest.raises(AssertionError):
            conn._coord_rank0_handle_new_load("d1", _load_meta())


def _pull_frames(*,
                 ranks=(0, ),
                 num_blocks=2,
                 num_layers=_NUM_LAYERS,
                 with_header=True,
                 uuid_echo=b"11"):
    """Construct a producer PULL response frame sequence with patterned per-layer test data."""
    frames = [
        _frame(zsb._MSG_OK),
        _frame(uuid_echo),
        _frame(str(len(ranks)).encode())
    ]
    if with_header:
        frames.append(_frame(zsb._secure_dumps({"tp_size": 1})))
    for rank in ranks:
        frames.append(_frame(str(rank).encode()))
        frames.extend(
            _frame(_layer_payload(num_blocks, 0xA0 + layer))
            for layer in range(num_layers))
    return frames


class TestRank0Pull:

    def _prepare(self, conn, sockets):
        """Configure make_zmq_socket to return the provided _FakeSocket instances in sequence."""
        handed = iter(sockets)
        return patch(f"{_BASE}.make_zmq_socket",
                     side_effect=lambda **kw: next(handed))

    def test_successful_pull_unpacks_and_broadcasts_load_notify(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_rank_to_identity = {1: b"r1"}
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11)
        sock = _FakeSocket(inbound=[_pull_frames()])
        with self._prepare(conn, [sock]), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert entry.pull_ok
        assert entry.load_complete.is_set()
        assert conn._coord_pool.unpacked == [(0, 0, 2, _NUM_LAYERS)]
        # Verify serialized payload bytes reached the shm slot in sequential
        # layer order.
        landed = conn._coord_pool.rank_layer_views(0, 0, 2)
        assert [bytes(v) for v in landed] == [
            _layer_payload(2, 0xA0 + layer) for layer in range(_NUM_LAYERS)
        ]
        assert sock.closed
        assert sock.sent[0][0] == zsb._MSG_PULL
        [(_ident, tag, notify)] = _drain_ipc(conn)
        assert (tag, notify) == (zsb._IPC_LOAD_NOTIFY, (11, 0, 2, [0, 1]))

    def test_multi_host_pull_uses_one_socket_per_host_and_channel(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn,
                            uuid=11,
                            remote_host=["h0", "h1"],
                            remote_port=[9100, 9200])
        socks = [
            _FakeSocket(inbound=[_pull_frames(ranks=())]),
            _FakeSocket(inbound=[_pull_frames(ranks=(0, ))]),
        ]
        with self._prepare(conn, socks), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert entry.pull_ok
        assert len(conn._coord_pool.unpacked) == 1
        assert all(s.closed for s in socks)

    def test_parallel_channels_each_carry_their_own_ranks(self):
        conn = _attach_runner(_make_conn(is_producer=False, tp_size=2),
                              ranks_per_host=2)
        conn._coord_local_to_global_rank = {0: 0, 1: 1}
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11)
        socks = [
            _FakeSocket(inbound=[_pull_frames(ranks=(0, ))]),
            _FakeSocket(
                inbound=[_pull_frames(ranks=(1, ), with_header=False)]),
        ]
        with self._prepare(conn, socks), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert entry.pull_ok
        assert sorted(conn._coord_pool.unpacked) == [(0, 0, 2, _NUM_LAYERS),
                                                     (0, 1, 2, _NUM_LAYERS)]

    def test_errors_on_every_channel_fail_the_pull_once(self):
        conn = _attach_runner(_make_conn(is_producer=False, tp_size=2),
                              ranks_per_host=2)
        conn._coord_channel_executor = _InlineExecutor()
        conn._coord_rank_to_identity = {1: b"r1"}
        entry = _recv_entry(conn, uuid=11, req_id="d1")
        socks = [
            _FakeSocket(inbound=[[_frame(zsb._MSG_ERR)]]),
            _FakeSocket(inbound=[[_frame(zsb._MSG_ERR)]]),
        ]
        with self._prepare(conn, socks), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert not entry.pull_ok
        assert all(s.closed for s in socks)
        # Invariant: Channel errors are deduplicated so that exactly one IPC_DROP
        # is sent and the shm slot is released only once regardless of channel error count.
        assert [m[1] for m in _drain_ipc(conn)] == [zsb._IPC_DROP]
        assert conn._coord_pool.released == [0]
        assert 11 not in conn._coord_recv
        assert conn._coord_done_recving == {"d1"}

    def test_a_channel_retries_polling_until_the_reply_lands(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11)
        sock = _FakeSocket(inbound=[_pull_frames()], empty_polls=2)
        with self._prepare(conn, [sock]), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=5):
            conn._coord_rank0_pull(entry)
        assert entry.pull_ok
        assert sock.polls == 3

    def test_ranks_not_owned_by_this_host_are_skipped(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11)
        sock = _FakeSocket(inbound=[_pull_frames(ranks=(0, 5))])
        with self._prepare(conn, [sock]), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert entry.pull_ok
        assert conn._coord_pool.unpacked == [(0, 0, 2, _NUM_LAYERS)]

    def test_error_reply_drops_the_request(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_rank_to_identity = {1: b"r1"}
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11)
        sock = _FakeSocket(inbound=[[_frame(zsb._MSG_ERR), _frame(b"11")]])
        with self._prepare(conn, [sock]), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert not entry.pull_ok
        assert 11 not in conn._coord_recv
        assert conn._coord_done_recving == {"d0"}
        assert conn._coord_pool.released == [0]
        [(_ident, tag, _obj)] = _drain_ipc(conn)
        assert tag == zsb._IPC_DROP

    def test_timeout_marks_the_pull_failed(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11)
        with self._prepare(conn, [_FakeSocket()]), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=0):
            conn._coord_rank0_pull(entry)
        assert not entry.pull_ok

    def test_duplicate_rank_delivery_is_rejected(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11)
        sock = _FakeSocket(inbound=[_pull_frames(ranks=(0, 0))])
        with self._prepare(conn, [sock]), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert not entry.pull_ok

    def test_missing_channel_zero_header_is_rejected(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11)
        sock = _FakeSocket(inbound=[_pull_frames(with_header=False)])
        with self._prepare(conn, [sock]), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert not entry.pull_ok

    def test_an_empty_endpoint_list_is_rejected(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11, remote_host=[], remote_port=[])
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert not entry.pull_ok

    def test_short_rank_count_is_rejected(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_channel_executor = _InlineExecutor()
        entry = _recv_entry(conn, uuid=11)
        sock = _FakeSocket(inbound=[_pull_frames(ranks=())])
        with self._prepare(conn, [sock]), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_pull(entry)
        assert not entry.pull_ok


# ---------------------------------------------------------------------------
# Consumer: drain / scatter
# ---------------------------------------------------------------------------


class TestScatter:

    def test_empty_block_list_is_a_no_op(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_scatter_shard(0, 0, [])
        assert conn.sync_calls == []

    def test_naive_path_index_puts_every_layer(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        for layer_idx in range(_NUM_LAYERS):
            conn._coord_pool.layer_view(0, 0, layer_idx,
                                        2).fill_(layer_idx + 1.0)
        conn._coord_scatter_shard(0, 2, [4, 9])
        assert len(conn.sync_calls) == _NUM_LAYERS
        assert conn.fast_scatter_calls == []
        assert conn._lat._stats[zsb._LAT_H2D][0] == 1.0
        # Verify each layer's shard is transferred strictly into target block
        # indices without modifying other cache blocks.
        for layer_idx, cache in enumerate(conn.runner.kv_caches):
            assert torch.equal(
                cache[[4, 9]], torch.full((2, ) + _BLOCK_SHAPE,
                                          layer_idx + 1.0))
            assert torch.count_nonzero(cache) == 2 * cache[0].numel()

    def test_fast_path_replaces_the_runner_caches(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._kv_scatter_enabled = True
        originals = list(conn.runner.kv_caches)
        conn._coord_scatter_shard(0, 2, [0, 1])
        assert conn.fast_scatter_calls == [(_NUM_LAYERS, _NUM_LAYERS, [0, 1])]
        assert all(new is not old
                   for new, old in zip(conn.runner.kv_caches, originals))

    def test_scatter_and_ack_registers_the_copy_on_the_coordinator(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        with patch.object(conn, "_coord_rank0_register_copy") as reg:
            conn._coord_scatter_and_ack("d1", 11, 0, 2, [0, 1])
        reg.assert_called_once_with(11, 0)

    def test_scatter_and_ack_acks_over_ipc_on_other_ranks(self):
        conn = _attach_runner(
            _make_conn(is_producer=False, tp_rank=1, tp_size=2))
        conn._coord_scatter_and_ack("d1", 11, 0, 2, [0, 1])
        [(_ident, tag, ack)] = _drain_ipc(conn)
        assert (tag, ack) == (zsb._IPC_COPY_DONE, (11, 1))

    def test_scatter_failure_still_acks(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        with patch.object(conn, "_coord_scatter_shard",
                          side_effect=RuntimeError("hbm fault")), \
             patch.object(conn, "_coord_rank0_register_copy") as reg:
            conn._coord_scatter_and_ack("d1", 11, 0, 2, [0, 1])
        assert reg.called


class TestDrainScatter:

    def test_coordinator_scatters_a_completed_pull(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        entry = _recv_entry(conn, uuid=11)
        entry.pull_ok = True
        with patch.object(conn, "_coord_scatter_and_ack") as scatter:
            conn._coord_drain_scatter("d1", _load_meta(remote_blocks=None))
        scatter.assert_called_once_with("d1", 11, 0, 2, [0, 1])

    def test_coordinator_ignores_an_unknown_uuid(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        with patch.object(conn, "_coord_scatter_and_ack") as scatter:
            conn._coord_drain_scatter("d1", _load_meta(remote_blocks=None))
        assert not scatter.called

    def test_failed_pull_only_notifies_the_producer(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        _recv_entry(conn, uuid=11)
        with patch.object(conn, "_coord_rank0_send_notify") as notify, \
             patch.object(conn, "_coord_scatter_and_ack") as scatter:
            conn._coord_drain_scatter("d1", _load_meta(remote_blocks=None))
        assert notify.called and not scatter.called

    def test_worker_scatters_after_load_notify(self):
        conn = _attach_runner(
            _make_conn(is_producer=False, tp_rank=1, tp_size=2))
        conn._worker_pending_load[11] = (1, 3, [2, 3, 4])
        with patch.object(conn, "_coord_scatter_and_ack") as scatter:
            conn._coord_drain_scatter("d1", _load_meta(remote_blocks=None))
        scatter.assert_called_once_with("d1", 11, 1, 3, [2, 3, 4])

    def test_worker_skips_when_the_coordinator_dropped_the_request(self):
        conn = _attach_runner(
            _make_conn(is_producer=False, tp_rank=1, tp_size=2))
        conn._worker_pending_load[11] = None
        with patch.object(conn, "_coord_scatter_and_ack") as scatter:
            conn._coord_drain_scatter("d1", _load_meta(remote_blocks=None))
        assert not scatter.called


class TestWorkerWaitLoad:

    def test_returns_the_pending_entry(self):
        conn = _make_conn(tp_rank=1, tp_size=2)
        conn._worker_pending_load[3] = (1, 2, [0, 1])
        assert conn._coord_worker_wait_load(3) == (1, 2, [0, 1])
        assert 3 not in conn._worker_pending_load

    def test_skip_sentinel_returns_none_immediately(self):
        conn = _make_conn(tp_rank=1, tp_size=2)
        conn._worker_pending_load[3] = None
        assert conn._coord_worker_wait_load(3, timeout=5.0) is None

    def test_timeout_returns_none(self):
        conn = _make_conn(tp_rank=1, tp_size=2)
        assert conn._coord_worker_wait_load(3, timeout=0.01) is None


class TestRegisterCopy:

    def test_partial_copies_keep_the_slot(self):
        conn = _attach_runner(_make_conn(is_producer=False, tp_size=2))
        entry = _recv_entry(conn, uuid=11)
        conn._coord_rank0_register_copy(11, 0)
        assert entry.copied == {0}
        assert 11 in conn._coord_recv
        assert conn._coord_pool.released == []

    def test_last_copy_notifies_and_releases(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        _recv_entry(conn, uuid=11, slot_idx=1)
        with patch.object(conn, "_coord_rank0_send_notify") as notify:
            conn._coord_rank0_register_copy(11, 0)
        assert notify.called
        assert 11 not in conn._coord_recv
        assert conn._coord_pool.released == [1]

    def test_slot_is_released_even_if_notify_raises(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        _recv_entry(conn, uuid=11, slot_idx=1)
        with patch.object(conn, "_coord_rank0_send_notify",
                          side_effect=RuntimeError("no route")), \
             pytest.raises(RuntimeError):
            conn._coord_rank0_register_copy(11, 0)
        assert conn._coord_pool.released == [1]

    def test_unknown_uuid_is_ignored(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_rank0_register_copy(999, 0)


class TestSendNotify:

    def _consumer(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_notif_sockets = {}
        conn._coord_sockets_lock = threading.Lock()
        return conn

    def test_creates_and_caches_one_socket_per_host(self):
        conn = self._consumer()
        sock = _FakeSocket()
        with patch(f"{_BASE}.make_zmq_socket", return_value=sock) as factory:
            conn._coord_rank0_send_notify(_load_meta(side_port=9700), 11)
            conn._coord_rank0_send_notify(_load_meta(side_port=9700), 12)
        assert factory.call_count == 1
        assert sock.sent == ["11", "12"]
        assert list(conn._coord_notif_sockets) == ["tcp://10.0.0.2:9700"]

    def test_one_socket_per_host_in_a_multi_host_reply(self):
        conn = self._consumer()
        socks = [_FakeSocket(), _FakeSocket()]
        handed = iter(socks)
        meta = _load_meta(host=["h0", "h1"], port=[1, 2], side_port=9700)
        with patch(f"{_BASE}.make_zmq_socket",
                   side_effect=lambda **kw: next(handed)):
            conn._coord_rank0_send_notify(meta, 11)
        assert sorted(
            conn._coord_notif_sockets) == ["tcp://h0:9700", "tcp://h1:9701"]

    def test_falls_back_to_the_local_side_channel_base(self):
        conn = self._consumer()
        with patch(f"{_BASE}.make_zmq_socket", return_value=_FakeSocket()), \
             patch(f"{_BASE}.dist_utils.get_side_channel_port",
                   return_value="9600"):
            conn._coord_rank0_send_notify(_load_meta(), 11)
        assert list(conn._coord_notif_sockets) == ["tcp://10.0.0.2:9600"]


# ---------------------------------------------------------------------------
# IPC listener loops
# ---------------------------------------------------------------------------


def _router_frame(ident: bytes, tag: bytes, payload) -> list[bytes]:
    return [ident, tag, zsb._secure_dumps(payload)]


def _dealer_frame(tag: bytes, payload) -> list[bytes]:
    return [tag, zsb._secure_dumps(payload)]


class TestRank0IpcLoop:

    def _run(self, conn, inbound):
        sock = _FakeSocket(inbound=inbound, on_exhausted=conn._stop_event.set)
        conn._coord_ipc_sock = sock
        conn._coord_rank0_ipc_loop()
        return sock

    def test_hello_registers_identities_and_unblocks_broadcasts(self):
        conn = _attach_runner(_make_conn(tp_size=2))
        self._run(conn, [_router_frame(b"r1", zsb._IPC_HELLO, (1, 1))])
        assert conn._coord_rank_to_identity == {1: b"r1"}
        assert conn._coord_local_to_global_rank == {0: 0, 1: 1}
        assert conn._coord_workers_ready.is_set()

    def test_hello_from_only_some_ranks_keeps_broadcasts_gated(self):
        conn = _attach_runner(_make_conn(tp_size=3))
        self._run(conn, [_router_frame(b"r1", zsb._IPC_HELLO, (1, 1))])
        assert conn._coord_rank_to_identity == {1: b"r1"}
        assert not conn._coord_workers_ready.is_set()

    def test_stage_done_completes_the_entry(self):
        conn = _attach_runner(_make_conn(tp_size=2))
        entry = _send_entry(conn, uuid=7)
        entry.staged.add(0)
        self._run(conn,
                  [_router_frame(b"r1", zsb._IPC_STAGE_DONE, (7, 1, False))])
        assert entry.stage_complete.is_set()
        assert entry.staged_events[1].is_set()

    def test_stage_done_before_all_ranks_leaves_completion_clear(self):
        conn = _attach_runner(_make_conn(tp_size=4))
        entry = _send_entry(conn, uuid=7)
        self._run(conn,
                  [_router_frame(b"r1", zsb._IPC_STAGE_DONE, (7, 1, True))])
        assert entry.stage_failed
        assert not entry.stage_complete.is_set()

    def test_stage_done_for_an_unknown_uuid_is_dropped(self):
        conn = _attach_runner(_make_conn(tp_size=2))
        self._run(conn,
                  [_router_frame(b"r1", zsb._IPC_STAGE_DONE, (99, 1, False))])

    def test_stage_done_from_an_out_of_range_rank_is_counted(self):
        # TODO(#771): Rank identifiers in STAGE_DONE payloads are not validated
        # against [0, ranks_per_host), allowing out-of-bounds rank numbers to count
        # toward stage_complete even though per-rank events remain bounds-checked.
        # This test pins existing behavior until strict rank validation is added.
        conn = _attach_runner(_make_conn(tp_size=2))
        entry = _send_entry(conn, uuid=7)
        self._run(conn,
                  [_router_frame(b"r1", zsb._IPC_STAGE_DONE, (7, 9, False))])
        assert entry.staged == {9}
        assert not entry.stage_complete.is_set()
        assert not any(ev.is_set() for ev in entry.staged_events)

    def test_copy_done_from_every_rank_notifies_and_releases(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        _recv_entry(conn, uuid=11, slot_idx=1)
        with patch.object(conn, "_coord_rank0_send_notify") as notify:
            self._run(conn,
                      [_router_frame(b"r1", zsb._IPC_COPY_DONE, (11, 0))])
        assert notify.called
        assert conn._coord_pool.released == [1]

    def test_copy_done_short_of_the_rank_count_holds_the_slot(self):
        conn = _attach_runner(_make_conn(is_producer=False, tp_size=2))
        entry = _recv_entry(conn, uuid=11)
        self._run(conn, [_router_frame(b"r1", zsb._IPC_COPY_DONE, (11, 1))])
        assert entry.copied == {1}
        assert conn._coord_pool.released == []

    def test_copy_done_for_an_unknown_uuid_is_dropped(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        self._run(conn, [_router_frame(b"r1", zsb._IPC_COPY_DONE, (99, 0))])

    def test_unknown_tag_is_logged(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [_router_frame(b"r1", b"NOPE", (1, ))])
        assert log.warning.called

    def test_malformed_frame_count_is_dropped(self):
        conn = _attach_runner(_make_conn())
        self._run(conn, [[b"r1", b"only-two"]])

    def test_unauthenticated_payload_is_dropped(self):
        conn = _attach_runner(_make_conn())
        self._run(conn, [[b"r1", zsb._IPC_HELLO, b"garbage"]])
        assert conn._coord_rank_to_identity == {}

    def test_zmq_error_is_logged_and_the_loop_continues(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [zmq.ZMQError("EAGAIN")])
        assert log.warning.called

    def test_context_termination_returns(self):
        conn = _attach_runner(_make_conn())
        sock = _FakeSocket(inbound=[zmq.ContextTerminated()])
        conn._coord_ipc_sock = sock
        conn._coord_rank0_ipc_loop()
        assert not conn._stop_event.is_set()

    def test_outbound_queue_is_drained_each_pass(self):
        conn = _attach_runner(_make_conn())
        conn._coord_ipc_out.put((b"r1", b"T", b"payload"))
        sock = self._run(conn, [])
        assert sock.sent == [[b"r1", b"T", b"payload"]]


class TestWorkerIpcLoop:

    def _worker(self, tp_size=2):
        return _attach_runner(_make_conn(tp_rank=1, tp_size=tp_size))

    def _run(self, conn, inbound):
        sock = _FakeSocket(inbound=inbound, on_exhausted=conn._stop_event.set)
        conn._coord_ipc_sock = sock
        conn._coord_worker_ipc_loop()
        return sock

    def test_stage_notify_lands_in_pending_stage(self):
        conn = self._worker()
        self._run(conn,
                  [_dealer_frame(zsb._IPC_STAGE_NOTIFY, (5, 1, 3, [7, 8, 9]))])
        assert conn._worker_pending_stage[5] == (1, 3, [7, 8, 9])

    def test_load_notify_four_tuple(self):
        conn = self._worker()
        self._run(conn,
                  [_dealer_frame(zsb._IPC_LOAD_NOTIFY, (5, 1, 2, [0, 1]))])
        assert conn._worker_pending_load[5] == (1, 2, [0, 1])

    def test_load_notify_five_tuple_carries_a_req_id(self):
        conn = self._worker()
        self._run(
            conn,
            [_dealer_frame(zsb._IPC_LOAD_NOTIFY, (5, "req", 1, 2, [0, 1]))])
        assert conn._worker_pending_load[5] == (1, 2, [0, 1])

    def test_load_skip_stores_the_none_sentinel(self):
        conn = self._worker()
        self._run(conn, [_dealer_frame(zsb._IPC_LOAD_SKIP, (5, ))])
        assert conn._worker_pending_load[5] is None

    def test_drop_clears_pending_stage_and_skips_the_load(self):
        conn = self._worker()
        conn._worker_pending_stage[5] = (0, 1, [0])
        self._run(conn, [_dealer_frame(zsb._IPC_DROP, (5, ))])
        assert conn._worker_pending_load[5] is None
        assert 5 not in conn._worker_pending_stage

    def test_unknown_tag_is_logged(self):
        conn = self._worker()
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [_dealer_frame(b"NOPE", (1, ))])
        assert log.warning.called

    def test_malformed_frame_count_is_dropped(self):
        conn = self._worker()
        self._run(conn, [[b"only-one"]])

    def test_unauthenticated_payload_is_dropped(self):
        conn = self._worker()
        self._run(conn, [[zsb._IPC_LOAD_SKIP, b"garbage"]])
        assert conn._worker_pending_load == {}

    def test_zmq_error_is_logged_and_the_loop_continues(self):
        conn = self._worker()
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [zmq.ZMQError("EAGAIN")])
        assert log.warning.called

    def test_context_termination_returns(self):
        conn = self._worker()
        conn._coord_ipc_sock = _FakeSocket(inbound=[zmq.ContextTerminated()])
        conn._coord_worker_ipc_loop()
        assert not conn._stop_event.is_set()

    def test_outbound_queue_is_drained_each_pass(self):
        conn = self._worker()
        conn._coord_ipc_out.put((None, b"T", b"payload"))
        sock = self._run(conn, [])
        assert sock.sent == [[b"T", b"payload"]]


# ---------------------------------------------------------------------------
# Producer: external data server and side channel
# ---------------------------------------------------------------------------


class TestExternalDataLoop:

    def _run(self, conn, inbound, *, response=None, send_error=None):
        sock = _FakeSocket(inbound=inbound,
                           on_exhausted=conn._stop_event.set,
                           send_error=send_error)
        conn._coord_channel_executor = _InlineExecutor()
        ctx = patch(f"{_BASE}.make_zmq_socket", return_value=sock)
        timeout = patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                        return_value=1)
        if response is None:
            with ctx, timeout:
                conn._coord_rank0_external_data_loop(0)
        else:
            with ctx, timeout, patch.object(
                    conn, "_coord_rank0_build_pull_response", **response):
                conn._coord_rank0_external_data_loop(0)
        return sock

    def test_serves_a_pull_with_the_prepared_frames(self):
        conn = _attach_runner(_make_conn())
        frames = [zsb._MSG_OK, b"7", b"1"]
        sock = self._run(conn, [[b"client", zsb._MSG_PULL, b"7", b"blocks"]],
                         response={"return_value": frames})
        assert sock.sent == [[b"client", *frames]]

    def test_build_failure_answers_err(self):
        conn = _attach_runner(_make_conn())
        sock = self._run(conn, [[b"client", zsb._MSG_PULL, b"7", b"b"]],
                         response={"side_effect": RuntimeError("no entry")})
        assert sock.sent == [[b"client", zsb._MSG_ERR, b"7"]]

    def test_none_response_answers_err(self):
        conn = _attach_runner(_make_conn())
        sock = self._run(conn, [[b"client", zsb._MSG_PULL, b"7", b"b"]],
                         response={"return_value": None})
        assert sock.sent == [[b"client", zsb._MSG_ERR, b"7"]]

    def test_send_errors_are_logged_not_raised(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [[b"client", zsb._MSG_PULL, b"7", b"b"]],
                      response={"return_value": [zsb._MSG_OK]},
                      send_error=zmq.ZMQError("EHOSTUNREACH"))
        assert log.warning.called

    def test_err_send_errors_are_logged_not_raised(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [[b"client", zsb._MSG_PULL, b"7", b"b"]],
                      response={"return_value": None},
                      send_error=zmq.ZMQError("EHOSTUNREACH"))
        assert log.warning.called

    def test_non_numeric_uuid_answers_bad_uuid(self):
        conn = _attach_runner(_make_conn())
        sock = self._run(conn, [[b"client", zsb._MSG_PULL, b"not-a-number"]])
        assert sock.sent == [[b"client", zsb._MSG_ERR, b"bad-uuid"]]

    def test_bad_uuid_send_error_is_logged(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [[b"client", zsb._MSG_PULL, b"nope"]],
                      send_error=zmq.ZMQError("EHOSTUNREACH"))
        assert log.warning.called

    def test_malformed_request_is_dropped(self):
        conn = _attach_runner(_make_conn())
        sock = self._run(conn, [[b"client", b"NOTPULL", b"7"]])
        assert sock.sent == []

    def test_zmq_error_is_logged_and_the_loop_continues(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [zmq.ZMQError("EAGAIN")])
        assert log.warning.called

    def test_context_termination_returns(self):
        conn = _attach_runner(_make_conn())
        sock = _FakeSocket(inbound=[zmq.ContextTerminated()])
        conn._coord_channel_executor = _InlineExecutor()
        with patch(f"{_BASE}.make_zmq_socket", return_value=sock), \
             patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_external_data_loop(0)
        assert not conn._stop_event.is_set()


class TestBuildPullResponse:

    def test_unknown_uuid_returns_none(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=0):
            assert conn._coord_rank0_build_pull_response(7, b"7", 0,
                                                         [0]) is None

    def test_a_pull_that_beats_its_send_metadata_backs_off_and_retries(self):
        conn = _attach_runner(_make_conn())

        def register_late():
            time.sleep(0.15)
            entry = _send_entry(conn, uuid=7, num_blocks=2)
            entry.staged_events[0].set()

        thread = threading.Thread(target=register_late)
        thread.start()
        try:
            with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                       return_value=5):
                frames = conn._coord_rank0_build_pull_response(7, b"7", 0, [0])
        finally:
            thread.join()
        assert frames[0] == zsb._MSG_OK

    def test_channel_zero_prepends_the_header(self):
        conn = _attach_runner(_make_conn())
        entry = _send_entry(conn, uuid=7, num_blocks=2)
        entry.staged_events[0].set()
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            frames = conn._coord_rank0_build_pull_response(7, b"7", 0, [0])
        assert frames[0] == zsb._MSG_OK
        assert frames[1] == b"7"
        assert frames[2] == b"1"
        assert zsb._secure_loads(frames[3])["num_layers"] == _NUM_LAYERS
        assert frames[4] == b"0"
        assert len(frames) == 5 + _NUM_LAYERS

    def test_other_channels_omit_the_header(self):
        conn = _attach_runner(_make_conn(tp_size=2), ranks_per_host=2)
        conn._coord_local_to_global_rank = {0: 0, 1: 1}
        entry = _send_entry(conn, uuid=7, num_blocks=2)
        entry.staged_events[1].set()
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            frames = conn._coord_rank0_build_pull_response(7, b"7", 1, [1])
        assert frames[3] == b"1"
        assert len(frames) == 4 + _NUM_LAYERS

    def test_pull_refreshes_the_expiration_deadline(self):
        conn = _attach_runner(_make_conn())
        entry = _send_entry(conn, uuid=7, ttl=-100.0)
        entry.staged_events[0].set()
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            conn._coord_rank0_build_pull_response(7, b"7", 0, [0])
        assert entry.expiration_time > time.perf_counter()

    def test_stage_timeout_returns_none(self):
        conn = _attach_runner(_make_conn())
        _send_entry(conn, uuid=7)
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=0.01):
            assert conn._coord_rank0_build_pull_response(7, b"7", 0,
                                                         [0]) is None

    def test_stage_failure_returns_none(self):
        conn = _attach_runner(_make_conn())
        entry = _send_entry(conn, uuid=7)
        entry.staged_events[0].set()
        entry.stage_failed = True
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            assert conn._coord_rank0_build_pull_response(7, b"7", 0,
                                                         [0]) is None

    def test_out_of_range_rank_falls_back_to_the_global_event(self):
        conn = _attach_runner(_make_conn(), ranks_per_host=4)
        entry = _send_entry(conn, uuid=7)
        entry.stage_complete.set()
        conn._coord_local_to_global_rank[3] = 3
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            frames = conn._coord_rank0_build_pull_response(7, b"7", 0, [3])
        assert frames[4] == b"3"

    def test_out_of_range_rank_can_also_time_out(self):
        conn = _attach_runner(_make_conn())
        _send_entry(conn, uuid=7)
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=0.01):
            assert conn._coord_rank0_build_pull_response(7, b"7", 0,
                                                         [3]) is None


class TestExternalNotifLoop:

    def _run(self, conn, inbound):
        sock = _FakeSocket(inbound=inbound, on_exhausted=conn._stop_event.set)
        with patch(f"{_BASE}.make_zmq_socket", return_value=sock):
            conn._coord_rank0_external_notif_loop()
        return sock

    def test_ack_marks_the_send_entry(self):
        conn = _attach_runner(_make_conn())
        entry = _send_entry(conn, uuid=7)
        self._run(conn, [[b"client", b"7"]])
        assert entry.pull_acked

    def test_stray_ack_is_logged(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [[b"client", b"99"]])
        assert log.warning.called

    def test_non_numeric_uuid_is_dropped(self):
        conn = _attach_runner(_make_conn())
        self._run(conn, [[b"client", b"nope"]])

    def test_zmq_error_is_logged_and_the_loop_continues(self):
        conn = _attach_runner(_make_conn())
        with patch(f"{_BASE}.logger") as log:
            self._run(conn, [zmq.ZMQError("EAGAIN")])
        assert log.warning.called

    def test_context_termination_returns(self):
        conn = _attach_runner(_make_conn())
        sock = _FakeSocket(inbound=[zmq.ContextTerminated()])
        with patch(f"{_BASE}.make_zmq_socket", return_value=sock):
            conn._coord_rank0_external_notif_loop()
        assert not conn._stop_event.is_set()


class TestExpireLoop:

    def test_expired_entries_release_their_slot(self):
        conn = _attach_runner(_make_conn())
        _send_entry(conn, uuid=7, req_id="r7", slot_idx=1, ttl=-1.0)
        conn._stop_event = _ScriptedEvent([False])
        conn._coord_rank0_expire_loop()
        assert conn._coord_done_sending == {"r7"}
        assert conn._coord_pool.released == [1]
        assert 7 not in conn._coord_send

    def test_acked_entries_are_left_for_get_finished(self):
        conn = _attach_runner(_make_conn())
        entry = _send_entry(conn, uuid=7, ttl=-1.0)
        entry.pull_acked = True
        conn._stop_event = _ScriptedEvent([False])
        conn._coord_rank0_expire_loop()
        assert 7 in conn._coord_send
        assert conn._coord_pool.released == []

    def test_an_entry_removed_mid_sweep_is_skipped(self):
        conn = _attach_runner(_make_conn())
        conn._coord_send = _VanishingDict()
        _send_entry(conn, uuid=7, req_id="r7", slot_idx=1, ttl=-1.0)
        conn._stop_event = _ScriptedEvent([False])
        conn._coord_rank0_expire_loop()
        # Double-release defense: entries already reported by get_finished() must
        # not be released a second time during the expiration sweep.
        assert conn._coord_pool.released == []
        assert conn._coord_done_sending == set()

    def test_a_set_stop_event_returns_immediately(self):
        conn = _attach_runner(_make_conn())
        conn._stop_event.set()
        conn._coord_rank0_expire_loop()


# ---------------------------------------------------------------------------
# get_finished and stats
# ---------------------------------------------------------------------------


class TestGetFinished:

    def test_completed_loads_are_reported_exactly_once(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        entry = _recv_entry(conn, uuid=11, req_id="d11")
        entry.pull_ok = True
        entry.load_complete.set()
        assert conn._coord_get_finished()[1] == {"d11"}
        assert entry.reported_done
        # Retain the receive entry in _coord_recv until all worker ranks
        # acknowledge completion via IPC_COPY_DONE.
        assert conn._coord_get_finished()[1] == set()
        assert 11 in conn._coord_recv

    def test_failure_surfacing_reqs_are_merged_in(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        conn._coord_done_recving.add("failed")
        assert conn._coord_get_finished()[1] == {"failed"}
        assert conn._coord_done_recving == set()


class TestConnectorStats:

    def test_producer_records_the_prefill_queue_length(self):
        conn = _attach_runner(_make_conn())
        _send_entry(conn, uuid=7)
        stats = conn.get_kv_connector_stats()
        assert stats is not None
        assert stats.data["prefill_queue_length"] == [1]

    def test_consumer_records_the_decode_queue_length(self):
        conn = _attach_runner(_make_conn(is_producer=False))
        _recv_entry(conn, uuid=11)
        stats = conn.get_kv_connector_stats()
        assert stats is not None
        assert stats.data["decode_queue_length"] == [1]

    def test_samples_are_cleared_after_being_read(self):
        conn = _attach_runner(_make_conn())
        conn.transfer_stats.record_d2h_transfer(1.5)
        assert conn.get_kv_connector_stats().data["d2h_transfer_time"] == [1.5]
        assert conn.get_kv_connector_stats().data["d2h_transfer_time"] == []

    def test_non_coordinator_reports_only_its_own_counters(self):
        conn = _attach_runner(_make_conn(tp_rank=1, tp_size=2))
        assert conn.get_kv_connector_stats() is None
        conn.transfer_stats.record_d2h_transfer(1.0)
        assert conn.get_kv_connector_stats() is not None


# ---------------------------------------------------------------------------
# End-to-end wiring across the coordinator and one worker rank
# ---------------------------------------------------------------------------


class TestCoordinatorWorkerHandoff:
    """Integration test driving coordinator and worker ranks through one request.

    The two connectors exchange IPC payloads directly in-memory, exercising the
    complete sequence (STAGE_NOTIFY -> stage -> STAGE_DONE -> PULL) without live
    network sockets.
    """

    def test_stage_notify_round_trip_serves_a_pull(self):
        coord = _attach_runner(_make_conn(tp_size=2), ranks_per_host=2)
        worker = _attach_runner(_make_conn(tp_rank=1, tp_size=2),
                                ranks_per_host=2)
        coord._coord_local_to_global_rank = {0: 0, 1: 1}
        coord._coord_rank_to_identity = {1: b"tpu_rank_1"}

        meta = SendMeta(uuid=42,
                        local_block_ids=[0, 1],
                        expiration_time=time.perf_counter() + 30)
        coord._coord_rank0_handle_new_send("req-a", meta)

        # Forward IPC_STAGE_NOTIFY to the worker, then trigger local staging.
        [(_ident, tag, notify)] = _drain_ipc(coord)
        assert tag == zsb._IPC_STAGE_NOTIFY
        uuid, slot_idx, num_blocks, block_ids = notify
        with worker._worker_pending_stage_cv:
            worker._worker_pending_stage[uuid] = (slot_idx, num_blocks,
                                                  block_ids)
        worker._coord_worker_stage_inline(uuid, "req-a", timeout=1.0)
        worker._coord_stage_wait_worker(worker._stage_pending_q.get_nowait())

        [(_ident, tag, done)] = _drain_ipc(worker)
        assert tag == zsb._IPC_STAGE_DONE
        done_uuid, rank, failed = done
        assert not failed

        # Complete rank 0 local D2H transfer and verify all ranks finish staging.
        coord._coord_stage_wait_worker(coord._stage_pending_q.get_nowait())
        sock = _FakeSocket(inbound=[
            _router_frame(b"tpu_rank_1", zsb._IPC_STAGE_DONE,
                          (done_uuid, rank, failed))
        ],
                           on_exhausted=coord._stop_event.set)
        coord._coord_ipc_sock = sock
        coord._coord_rank0_ipc_loop()

        entry = coord._coord_send[42]
        assert entry.stage_complete.is_set()

        coord._stop_event.clear()
        with patch(f"{_BASE}.dist_utils.get_p2p_wait_pull_timeout",
                   return_value=1):
            frames = coord._coord_rank0_build_pull_response(
                42, b"42", 0, [0, 1])
        assert frames[2] == b"2"
        assert frames[4] == b"0"

    def test_copy_done_round_trip_releases_the_slot(self):
        coord = _attach_runner(_make_conn(is_producer=False, tp_size=2),
                               ranks_per_host=2)
        worker = _attach_runner(_make_conn(is_producer=False,
                                           tp_rank=1,
                                           tp_size=2),
                                ranks_per_host=2)
        coord._coord_notif_sockets = {}
        coord._coord_sockets_lock = threading.Lock()
        _recv_entry(coord, uuid=11, slot_idx=1, side_port=9700)

        # Scatter rank 0 local shard into device cache, then verify worker IPC_COPY_DONE ack.
        coord._coord_scatter_and_ack("d1", 11, 1, 2, [0, 1])
        assert 11 in coord._coord_recv

        worker._coord_scatter_and_ack("d1", 11, 1, 2, [0, 1])
        [(_ident, tag, ack)] = _drain_ipc(worker)
        assert tag == zsb._IPC_COPY_DONE

        sock = _FakeSocket(inbound=[_router_frame(b"tpu_rank_1", tag, ack)],
                           on_exhausted=coord._stop_event.set)
        coord._coord_ipc_sock = sock
        with patch(f"{_BASE}.make_zmq_socket", return_value=_FakeSocket()):
            coord._coord_rank0_ipc_loop()

        assert 11 not in coord._coord_recv
        assert coord._coord_pool.released == [1]
