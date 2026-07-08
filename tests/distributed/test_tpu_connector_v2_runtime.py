# SPDX-License-Identifier: Apache-2.0

import json
import socket
import sys
import threading
import types

import pytest

from .tpu_connector_v2_test_utils import (_assert_log_messages,
                                          _capture_logger_messages,
                                          _load_v2_module)


def test_v2_connector_requires_unified_block_pool(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    sys.modules[
        "vllm_torchtpu.envs"].TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL = False

    config = types.SimpleNamespace(kv_transfer_config=object())
    with pytest.raises(ValueError, match="UNIFIED_BLOCK_POOL"):
        mod.TPUConnectorV2(config, mod.KVConnectorRole.SCHEDULER)


def _strided_transfer_module():
    return sys.modules[
        "vllm_torchtpu.distributed.kv_transfer.v2.strided_transfer"]


def _registered_memory_region(**kwargs):
    return _strided_transfer_module().RegisteredMemoryRegion(**kwargs)


def _remote_worker_metadata(**kwargs):
    return _strided_transfer_module().RemoteWorkerMetadata(**kwargs)


def _pop_worker_completions(mod, worker):
    meta = worker.build_connector_worker_meta()
    assert isinstance(meta, mod.TPUConnectorV2WorkerMeta)
    assert worker.build_connector_worker_meta() is None
    return meta.completions


def _disable_v2_pull_start(worker):
    worker._request_v2_pull_start = lambda req_meta: None


def _blocks(*groups):

    class Blocks:

        @staticmethod
        def get_block_ids():
            return [list(group) for group in groups]

    return Blocks()


def _finished_length_capped_status():
    return types.SimpleNamespace(name="FINISHED_LENGTH_CAPPED")


def _v2_lifecycle_worker(mod):
    worker = mod.TPUConnectorV2Worker(object())
    worker.node_id = 0
    worker._coord_lock = threading.Lock()
    worker._coord_send = {}
    worker._coord_done_sending = set()
    worker._strided_timeout_s = lambda: 120.0
    return worker


def _local_test_host() -> str:
    return socket.gethostbyname("localhost")


def test_v2_zmq_side_channel_lives_outside_connector(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    zmq_side_channel = sys.modules[
        "vllm_torchtpu.distributed.kv_transfer.v2.zmq_side_channel"]

    assert "_send_v2_side_channel_request" not in vars(mod)
    assert callable(zmq_side_channel.send_request)
    assert callable(zmq_side_channel.serve_lifecycle_requests)


def _pick_strided_base_port() -> int:
    # The connector adds a 10000-port stride to kv_transfer_port.
    max_base_port = 65535 - 10000
    for _ in range(32):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = int(sock.getsockname()[1])
        if port <= max_base_port:
            return port
    raise RuntimeError("could not find an available low port for the test")


class _RecordingWriteSession:

    def __init__(self):
        self.flushes = 0
        self.discards = 0

    def flush_all(self):
        self.flushes += 1

    def discard(self):
        self.discards += 1


def test_v2_pull_start_extends_live_entry(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = _v2_lifecycle_worker(mod)
    entry = mod._V2SendLifecycleEntry(req_id="req-live",
                                      expiration_time=1005.0)
    worker._coord_send[10] = entry
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 1000.0)

    ok, message = worker._coord_rank0_handle_v2_pull_start(10)

    assert ok is True
    assert message == "ok"
    assert entry.pull_started is True
    assert entry.expiration_time == 1120.0
    assert worker._coord_done_sending == set()


def test_v2_pull_start_end_logs_producer_lifecycle(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = _v2_lifecycle_worker(mod)
    entry = mod._V2SendLifecycleEntry(req_id="req-live",
                                      expiration_time=1005.0)
    worker._coord_send[10] = entry
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 1000.0)

    with _capture_logger_messages(mod.logger) as messages:
        assert worker._coord_rank0_handle_v2_pull_start(10) == (True, "ok")
        assert worker._coord_rank0_handle_v2_pull_end(10) == (True, "ok")

    _assert_log_messages(messages, "recv START", "accept START", "recv END",
                         "accept END")


def test_v2_pull_start_end_logs_consumer_side_channel(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    sent = []

    def fake_send_request(**kwargs):
        sent.append(kwargs)
        return [mod.zmq_side_channel.MSG_OK]

    monkeypatch.setattr(mod.zmq_side_channel, "send_request",
                        fake_send_request)

    worker = mod.TPUConnectorV2Worker(object())
    worker._strided_timeout_s = lambda: 120.0
    req_meta = types.SimpleNamespace(uuid=123,
                                     v2_ack_host="10.0.0.9",
                                     v2_ack_port=7600)

    with _capture_logger_messages(mod.logger) as messages:
        worker._request_v2_pull_start(req_meta)

        scheduler = mod.TPUConnectorV2Scheduler(
            types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
                parallel_config=types.SimpleNamespace(data_parallel_rank=0),
            ))
        scheduler._send_v2_end(req_meta)

    assert [item["tag"] for item in sent] == [
        mod.zmq_side_channel.PULL_START,
        mod.zmq_side_channel.PULL_END,
    ]
    _assert_log_messages(messages, "send START", "START ack OK", "send END",
                         "END ack OK")


def test_v2_lifecycle_broadcasts_to_all_producer_ack_targets(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    dist_utils = sys.modules["vllm_torchtpu.distributed.utils"]
    monkeypatch.setattr(dist_utils, "get_node_id", lambda: 99)
    sent = []

    def fake_send_request(**kwargs):
        sent.append(kwargs)
        return [mod.zmq_side_channel.MSG_OK]

    monkeypatch.setattr(mod.zmq_side_channel, "send_request",
                        fake_send_request)

    req_meta = types.SimpleNamespace(uuid=123,
                                     v2_ack_host=("10.0.0.10", "10.0.0.11"),
                                     v2_ack_port=(9600, 9601))
    worker = mod.TPUConnectorV2Worker(object())
    worker._strided_timeout_s = lambda: 120.0
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))

    worker._request_v2_pull_start(req_meta)
    scheduler._send_v2_end(req_meta)

    assert [(item["tag"], item["host"], item["port"]) for item in sent] == [
        (mod.zmq_side_channel.PULL_START, "10.0.0.10", 9600),
        (mod.zmq_side_channel.PULL_START, "10.0.0.11", 9601),
        (mod.zmq_side_channel.PULL_END, "10.0.0.10", 9600),
        (mod.zmq_side_channel.PULL_END, "10.0.0.11", 9601),
    ]


def test_v2_producer_host_coordinator_registers_send_lifecycle(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    monkeypatch.setenv("TPU_NUM_HOSTS", "2")
    monkeypatch.setattr(mod, "_get_native_tp_rank", lambda: 4)
    monkeypatch.setattr(mod, "_get_native_tp_size", lambda: 8)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
        parallel_config=types.SimpleNamespace(data_parallel_rank=0,
                                              tensor_parallel_size=8),
    )
    worker = mod.TPUConnectorV2Worker(config)
    metadata = types.SimpleNamespace(
        reqs_to_send={
            "req-1": types.SimpleNamespace(uuid=456, expiration_time=1234.0)
        })

    worker._process_v2_sends(metadata)

    assert worker._is_host_coordinator
    assert 456 in worker._coord_send
    assert worker._coord_send[456].req_id == "req-1"


def test_v2_host_coordinator_reports_finished_sending(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    monkeypatch.setenv("TPU_NUM_HOSTS", "2")
    monkeypatch.setattr(mod, "_get_native_tp_rank", lambda: 4)
    monkeypatch.setattr(mod, "_get_native_tp_size", lambda: 8)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
        parallel_config=types.SimpleNamespace(data_parallel_rank=0,
                                              tensor_parallel_size=8),
    )
    worker = mod.TPUConnectorV2Worker(config)
    entry = mod._V2SendLifecycleEntry(req_id="req-1", expiration_time=1234.0)
    entry.pull_acked = True
    worker._coord_send[456] = entry

    assert worker._coord_get_finished() == ({"req-1"}, set())


def test_v2_multihost_ack_endpoints_match_producer_host_coordinators(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    monkeypatch.setenv("TPU_NUM_HOSTS", "2")
    monkeypatch.setattr(mod, "_get_native_tp_size", lambda: 8)
    dist_utils = sys.modules["vllm_torchtpu.distributed.utils"]
    remote_hosts = ("10.0.0.10", "10.0.0.11")
    endpoint_by_dp_node = {}

    for dp_rank in (0, 1):
        scheduler_config = types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
            parallel_config=types.SimpleNamespace(data_parallel_rank=dp_rank),
        )
        scheduler = mod.TPUConnectorV2Scheduler(scheduler_config)
        params = {"remote_host": remote_hosts}

        scheduler._attach_v2_side_channel_endpoint(params)

        for node_id, host_tp_rank in ((0, 0), (1, 4)):
            monkeypatch.setattr(dist_utils,
                                "get_node_id",
                                lambda node_id=node_id: node_id)
            monkeypatch.setattr(mod,
                                "_get_native_tp_rank",
                                lambda host_tp_rank=host_tp_rank: host_tp_rank)
            worker_config = types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
                parallel_config=types.SimpleNamespace(
                    data_parallel_rank=dp_rank, tensor_parallel_size=8),
            )
            worker = mod.TPUConnectorV2Worker(worker_config)
            uuid = 1000 + dp_rank * 10 + node_id
            metadata = types.SimpleNamespace(
                reqs_to_send={
                    f"req-{dp_rank}-{node_id}":
                    types.SimpleNamespace(uuid=uuid, expiration_time=1234.0)
                })

            worker._process_v2_sends(metadata)

            endpoint = (params["v2_ack_host"][node_id],
                        params["v2_ack_port"][node_id])
            endpoint_by_dp_node[(dp_rank, node_id)] = endpoint
            assert worker.side_channel_port == endpoint[1]
            assert uuid in worker._coord_send

    assert endpoint_by_dp_node[(0, 1)][1] == endpoint_by_dp_node[(1, 0)][1]
    assert endpoint_by_dp_node[(0, 1)] != endpoint_by_dp_node[(1, 0)]


def test_v2_worker_logs_strided_lifecycle_metadata(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingBridge:

        def __init__(self):
            self.transfer_engine = types.SimpleNamespace(
                register_other_remote_metadata=lambda metadata: None)
            self.sessions = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            return 1

        def new_destination_write_session(self):
            session = _RecordingWriteSession()
            self.sessions.append(session)
            return session

    op = mod.StridedSegmentOp(src_offset_bytes=0,
                              dst_offset_bytes=0,
                              segment_bytes=1,
                              src_stride_bytes=1,
                              dst_stride_bytes=1,
                              num_segments=1,
                              source_region_id="layer.0",
                              destination_region_id="layer.0",
                              layer_name="layer.0",
                              layer_type=mod.LayerType.FULL_ATTN)
    req_meta = types.SimpleNamespace(
        uuid=123,
        remote_block_ids=[1],
        remote_dp_rank=0,
        remote_metadata=(),
        remote_rank_ops_by_decode_rank={0: {
            0: (op, )
        }},
    )
    metadata = types.SimpleNamespace(reqs_to_send={},
                                     reqs_to_load={"req-1": req_meta})
    worker = mod.TPUConnectorV2Worker(object())
    worker.tp_rank = 0
    worker.set_strided_transfer_bridge(RecordingBridge())
    _disable_v2_pull_start(worker)

    with _capture_logger_messages(mod.logger) as messages:
        assert worker.process_send_load(metadata) == 1
        (completion, ) = _pop_worker_completions(mod, worker)

    assert completion.req_id == "req-1"
    _assert_log_messages(messages, "local START", "local END queued",
                         "END completion meta")


def test_v2_scheduler_logs_strided_lifecycle_metadata(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    req_meta = types.SimpleNamespace(uuid=456,
                                     remote_block_ids=[1],
                                     v2_ack_host="10.0.0.9",
                                     v2_ack_port=7600,
                                     remote_rank_ops_by_decode_rank={
                                         0: {},
                                         1: {},
                                     })
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))
    scheduler._send_v2_end = lambda meta: None

    with _capture_logger_messages(mod.logger) as messages:
        scheduler._mark_recv_lifecycle_planned(req_id="req-1",
                                               uuid=456,
                                               req_meta=req_meta,
                                               expected_tp_ranks={0, 1})
        scheduler._mark_recv_lifecycle_dispatched(req_id="req-1", uuid=456)
        output = types.SimpleNamespace(
            finished_recving=None,
            kv_connector_worker_meta=mod.TPUConnectorV2WorkerMeta(
                completions=(mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                                uuid=456,
                                                                dp_rank=0,
                                                                tp_rank=0,
                                                                success=True,
                                                                error=""), )))
        scheduler.update_connector_output(output)
        output.kv_connector_worker_meta = mod.TPUConnectorV2WorkerMeta(
            completions=(mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                            uuid=456,
                                                            dp_rank=0,
                                                            tp_rank=1,
                                                            success=True,
                                                            error=""), ))
        scheduler.update_connector_output(output)

    assert output.finished_recving == {"req-1"}
    _assert_log_messages(messages, "START planned", "START dispatched",
                         "recv END completion", "recv END complete")


def test_v2_pull_start_rejects_unknown_uuid(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = _v2_lifecycle_worker(mod)

    ok, message = worker._coord_rank0_handle_v2_pull_start(
        404, wait_for_register=False)

    assert ok is False
    assert "unknown uuid=404" in message


def test_v2_pull_start_rejects_expired_entry(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = _v2_lifecycle_worker(mod)
    entry = mod._V2SendLifecycleEntry(req_id="req-expired",
                                      expiration_time=999.0)
    worker._coord_send[11] = entry
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 1000.0)

    ok, message = worker._coord_rank0_handle_v2_pull_start(11)

    assert ok is False
    assert "expired uuid=11" in message
    assert 11 not in worker._coord_send
    assert worker._coord_done_sending == {"req-expired"}


def test_v2_pull_end_acks_live_started_entry(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = _v2_lifecycle_worker(mod)
    entry = mod._V2SendLifecycleEntry(req_id="req-live",
                                      expiration_time=1005.0)
    entry.pull_started = True
    worker._coord_send[12] = entry
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 1000.0)

    ok, message = worker._coord_rank0_handle_v2_pull_end(12)

    assert ok is True
    assert message == "ok"
    assert entry.pull_acked is True
    assert 12 in worker._coord_send
    assert worker._coord_done_sending == set()


def test_v2_pull_end_rejects_expired_started_entry(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = _v2_lifecycle_worker(mod)
    entry = mod._V2SendLifecycleEntry(req_id="req-expired",
                                      expiration_time=999.0)
    entry.pull_started = True
    worker._coord_send[13] = entry
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 1000.0)

    ok, message = worker._coord_rank0_handle_v2_pull_end(13)

    assert ok is False
    assert "expired uuid=13" in message
    assert 13 not in worker._coord_send
    assert worker._coord_done_sending == {"req-expired"}


def test_v2_pull_end_rejects_end_before_start(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = _v2_lifecycle_worker(mod)
    entry = mod._V2SendLifecycleEntry(req_id="req-not-started",
                                      expiration_time=1005.0)
    worker._coord_send[14] = entry
    monkeypatch.setattr(mod.time, "perf_counter", lambda: 1000.0)

    ok, message = worker._coord_rank0_handle_v2_pull_end(14)

    assert ok is False
    assert "end before start uuid=14" in message
    assert entry.pull_acked is False
    assert 14 in worker._coord_send


def test_v2_fa_block_ids_reject_missing_worker_group(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    grouped_block_ids = ((7, ), (7, ), (7, ))

    with pytest.raises(ValueError, match="block_ids has no group 3"):
        mod.TPUConnectorV2Scheduler._block_ids_for_fa_groups(
            grouped_block_ids, (3, ))


def test_v2_mamba_block_ids_reject_missing_worker_group(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    grouped_block_ids = ((7, ), (8, ), (9, ))

    with pytest.raises(ValueError, match="block_ids has no group 3"):
        mod.TPUConnectorV2Scheduler._block_ids_for_mamba_groups(
            grouped_block_ids, (3, ))


def test_v2_mamba_block_ids_preserve_distinct_group_ids(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    grouped_block_ids = ((7, ), (8, ), (9, ), (20, 21))

    assert mod.TPUConnectorV2Scheduler._block_ids_for_mamba_groups(
        grouped_block_ids, (0, 1, 2)) == (7, 8, 9)


def test_v2_mamba_block_ids_select_group_tail_slots(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    grouped_block_ids = (
        (0, 0, 0, 12),
        (0, 0, 0, 13),
        (0, 0, 0, 14),
        (9, 10, 15, 16),
    )

    assert mod.TPUConnectorV2Scheduler._block_ids_for_mamba_groups(
        grouped_block_ids, (0, 1, 2)) == (12, 13, 14)


def test_v2_worker_group_indices_require_group_is_mamba(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = object.__new__(mod.TPUConnectorV2Worker)
    worker._kv_cache_region_templates = {}

    with pytest.raises(RuntimeError, match="requires group_is_mamba"):
        worker._kv_cache_group_indices()


def test_v2_worker_group_indices_use_group_is_mamba(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = object.__new__(mod.TPUConnectorV2Worker)
    worker.group_is_mamba = (True, True, True, False)

    assert worker._kv_cache_group_indices() == ((3, ), (0, 1, 2))


def test_v2_source_metadata_preserves_block_ids_by_group(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(parallel_config=types.SimpleNamespace(
            tensor_parallel_rank=0)))
    source = _single_layer_source_metadata(mod)
    handshake = mod.TPUConnectorV2HandshakeMetadata(
        remote_metadata=_remote_worker_metadata(
            dp_rank=0,
            tp_rank=0,
            worker_id="p0",
            tcp_host="127.0.0.1",
            tcp_port=32100,
            transport="zmq",
            regions=(_registered_memory_region(region_id="layer.0",
                                               nbytes=64,
                                               page_bytes=16), ),
        ),
        kv_source_layout=source.kv_source_layout,
        kv_caches=source.kv_caches[0],
        topology=_single_layer_topology(mod),
        fa_group_indices=(3, ),
        mamba_group_indices=(0, 1, 2),
    )
    scheduler.set_xfer_handshake_metadata({0: handshake})

    grouped_block_ids = ((10, ), (11, ), (12, ), (20, 21))
    metadata = scheduler._producer_source_metadata(grouped_block_ids, {
        "uuid": 99,
    })

    assert metadata.block_ids_by_group == grouped_block_ids
    assert metadata.fa_block_ids == (20, 21)
    assert metadata.mamba_block_ids == (10, 11, 12)
    round_tripped = mod.ConnectorMetadataV2.from_mapping(metadata.to_dict())
    assert round_tripped.block_ids_by_group == grouped_block_ids


def test_v2_producer_metadata_uses_mamba_group_tail_slots(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(parallel_config=types.SimpleNamespace(
            tensor_parallel_rank=0)))
    source = _single_layer_source_metadata(mod)
    handshake = mod.TPUConnectorV2HandshakeMetadata(
        remote_metadata=_remote_worker_metadata(
            dp_rank=0,
            tp_rank=0,
            worker_id="p0",
            tcp_host="127.0.0.1",
            tcp_port=32100,
            transport="zmq",
            regions=(_registered_memory_region(region_id="layer.0",
                                               nbytes=64,
                                               page_bytes=16), ),
        ),
        kv_source_layout=source.kv_source_layout,
        kv_caches=source.kv_caches[0],
        topology=_single_layer_topology(mod),
        fa_group_indices=(3, ),
        mamba_group_indices=(0, 1, 2),
    )
    scheduler.set_xfer_handshake_metadata({0: handshake})

    grouped_block_ids = (
        (0, 0, 0, 12),
        (0, 0, 0, 13),
        (0, 0, 0, 14),
        (20, 21),
    )
    metadata = scheduler._producer_source_metadata(grouped_block_ids, {
        "uuid": 99,
    })

    assert metadata.mamba_block_ids == (12, 13, 14)
    assert metadata.mamba_num_tokens is None
    assert metadata.fa_block_ids == (20, 21)
    assert metadata.block_ids_by_group == (
        (12, ),
        (13, ),
        (14, ),
        (20, 21),
    )
    round_tripped = mod.ConnectorMetadataV2.from_mapping(metadata.to_dict())
    assert round_tripped.block_ids_by_group == metadata.block_ids_by_group


def test_v2_source_metadata_requires_grouped_block_ids(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    metadata = _single_layer_source_metadata(mod).to_dict()
    metadata.pop("block_ids_by_group")

    with pytest.raises(KeyError, match="block_ids_by_group"):
        mod.ConnectorMetadataV2.from_mapping(metadata)


def test_v2_local_decode_allocation_requires_grouped_block_ids(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    allocation = {
        "rank": 0,
        "block_size": 4,
        "kv_caches": {
            "layer.0":
            _single_layer_destination(mod).kv_caches["layer.0"].to_dict(),
        },
        "fa_block_ids": [7],
        "mamba_block_ids": [],
        "fa_num_tokens": 4,
        "mamba_num_tokens": None,
    }

    with pytest.raises(KeyError, match="block_ids_by_group"):
        mod.LocalDecodeAllocation.from_mapping(allocation)


def test_v2_strided_segment_op_requires_layer_identity(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    with pytest.raises(TypeError):
        mod.StridedSegmentOp(src_offset_bytes=0,
                             dst_offset_bytes=0,
                             segment_bytes=4,
                             src_stride_bytes=4,
                             dst_stride_bytes=4,
                             num_segments=1,
                             source_region_id="src",
                             destination_region_id="dst")


def test_v2_decode_metadata_rejects_short_block_id_group_view(monkeypatch, ):
    mod = _load_v2_module(monkeypatch)
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(parallel_config=types.SimpleNamespace(
            tensor_parallel_rank=0)))
    source = _single_layer_source_metadata(mod)
    handshake = mod.TPUConnectorV2HandshakeMetadata(
        remote_metadata=_remote_worker_metadata(
            dp_rank=0,
            tp_rank=0,
            worker_id="p0",
            tcp_host="127.0.0.1",
            tcp_port=32100,
            transport="zmq",
            regions=(_registered_memory_region(region_id="layer.0",
                                               nbytes=64,
                                               page_bytes=16), ),
        ),
        kv_source_layout=source.kv_source_layout,
        kv_caches=source.kv_caches[0],
        topology=_single_layer_topology(mod),
        fa_group_indices=(3, ),
        mamba_group_indices=(0, 1, 2),
    )
    scheduler.set_xfer_handshake_metadata({0: handshake})

    with pytest.raises(ValueError, match="block_ids has no group 3"):
        scheduler._decode_metadata(
            {
                "local_block_ids": ((7, ), (7, ), (7, )),
            },
            source,
        )


def test_strided_bridge_preserves_region_relative_transfer_ops(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingEngine:

        def __init__(self):
            self.calls = []

        def pull_from_registered_into_session(self, *, remote_tp_rank, ops,
                                              write_session, dp_rank):
            self.calls.append(
                (remote_tp_rank, tuple(ops), write_session, dp_rank))
            return sum(op.segment_bytes * op.num_segments for op in ops)

    engine = RecordingEngine()
    bridge = mod.TPUConnectorV2StridedBridge(engine)
    session = object()
    op = mod.StridedSegmentOp(src_offset_bytes=8,
                              dst_offset_bytes=16,
                              segment_bytes=4,
                              src_stride_bytes=12,
                              dst_stride_bytes=20,
                              num_segments=2,
                              source_region_id="source.layer",
                              destination_region_id="dest.layer",
                              layer_name="layer.0",
                              layer_type=mod.LayerType.FULL_ATTN)

    copied = bridge.pull_rank_ops_into_session(remote_tp_rank=3,
                                               ops=(op, ),
                                               write_session=session,
                                               dp_rank=2)

    assert copied == 8
    transfer_op = engine.calls[0][1][0]
    assert transfer_op.src_region_id == "source.layer"
    assert transfer_op.dst_region_id == "dest.layer"
    assert transfer_op.src_offset_bytes == 8
    assert transfer_op.dst_offset_bytes == 16
    assert transfer_op.src_stride_bytes == 12
    assert transfer_op.dst_stride_bytes == 20


def test_strided_bridge_primary_api_pulls_op_list(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingEngine:

        def __init__(self):
            self.calls = []

        def pull_from_registered_into_session(self, *, remote_tp_rank, ops,
                                              write_session, dp_rank):
            self.calls.append(
                (remote_tp_rank, tuple(ops), write_session, dp_rank))
            return sum(op.segment_bytes * op.num_segments for op in ops)

    engine = RecordingEngine()
    bridge = mod.TPUConnectorV2StridedBridge(engine)
    session = object()
    ops = (
        mod.StridedSegmentOp(
            src_offset_bytes=32,
            dst_offset_bytes=48,
            segment_bytes=4,
            src_stride_bytes=16,
            dst_stride_bytes=20,
            num_segments=3,
            source_region_id="source.layer",
            destination_region_id="dest.layer",
            layer_name="layer.0",
            layer_type=mod.LayerType.FULL_ATTN,
        ),
        mod.StridedSegmentOp(
            src_offset_bytes=80,
            dst_offset_bytes=96,
            segment_bytes=8,
            src_stride_bytes=8,
            dst_stride_bytes=8,
            num_segments=1,
            source_region_id="source.layer",
            destination_region_id="dest.layer",
            layer_name="layer.0",
            layer_type=mod.LayerType.FULL_ATTN,
        ),
    )

    copied = bridge.pull_rank_ops_into_session(remote_tp_rank=5,
                                               ops=ops,
                                               write_session=session,
                                               dp_rank=9)

    assert copied == 20
    assert len(engine.calls) == 1
    remote_tp_rank, transfer_ops, called_session, dp_rank = engine.calls[0]
    assert remote_tp_rank == 5
    assert called_session is session
    assert dp_rank == 9
    assert [(
        op.src_region_id,
        op.dst_region_id,
        op.src_offset_bytes,
        op.dst_offset_bytes,
        op.segment_bytes,
        op.src_stride_bytes,
        op.dst_stride_bytes,
        op.num_segments,
    ) for op in transfer_ops] == [
        ("source.layer", "dest.layer", 32, 48, 4, 16, 20, 3),
        ("source.layer", "dest.layer", 80, 96, 8, 8, 8, 1),
    ]


def test_worker_delegates_op_list_to_strided_bridge(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingBridge:

        def __init__(self):
            self.calls = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            self.calls.append(
                (remote_tp_rank, tuple(ops), write_session, dp_rank))
            return 321

    worker = mod.TPUConnectorV2Worker(object())
    bridge = RecordingBridge()
    worker.set_strided_transfer_bridge(bridge)
    session = object()
    ops = (mod.StridedSegmentOp(
        src_offset_bytes=4,
        dst_offset_bytes=8,
        segment_bytes=2,
        src_stride_bytes=6,
        dst_stride_bytes=10,
        num_segments=4,
        source_region_id="source.layer",
        destination_region_id="dest.layer",
        layer_name="layer.0",
        layer_type=mod.LayerType.FULL_ATTN,
    ), )

    copied = worker.pull_rank_ops_into_session(remote_tp_rank=6,
                                               ops=ops,
                                               write_session=session,
                                               dp_rank=3)

    assert copied == 321
    assert bridge.calls == [(6, ops, session, 3)]


def test_worker_registers_remote_metadata_from_connector_metadata(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    remote_metadata = (
        _remote_worker_metadata(dp_rank=0,
                                tp_rank=0,
                                worker_id="prefill-rank0",
                                tcp_host="10.0.0.1",
                                tcp_port=32100,
                                transport="zmq",
                                regions=()),
        _remote_worker_metadata(dp_rank=0,
                                tp_rank=1,
                                worker_id="prefill-rank1",
                                tcp_host="10.0.0.2",
                                tcp_port=32101,
                                transport="zmq",
                                regions=()),
    )

    class RecordingEngine:

        def __init__(self):
            self.registered = []

        def register_other_remote_metadata(self, metadata):
            self.registered.append(metadata)

    engine = RecordingEngine()
    worker = mod.TPUConnectorV2Worker(object())
    worker.set_strided_transfer_bridge(mod.TPUConnectorV2StridedBridge(engine))
    connector_metadata = types.SimpleNamespace(
        reqs_to_load={
            "req-1": types.SimpleNamespace(remote_metadata=remote_metadata),
            "req-2": types.SimpleNamespace(remote_metadata=()),
        })

    worker.register_remote_metadata_from_connector_metadata(connector_metadata)

    assert engine.registered == [remote_metadata]


def test_v2_validates_remote_region_metadata_for_source_kv_caches(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    metadata = mod.ConnectorMetadataV2(
        req_id=1,
        block_size=4,
        kv_source_layout=mod.KVParallelLayout(full_attn_pcp_size=1,
                                              full_attn_tp_size=1,
                                              linear_attn_pcp_size=1,
                                              linear_attn_tp_size=1),
        kv_caches={
            0: {
                "model.layers.0.self_attn":
                mod.KVCacheRegion(
                    layer_name="model.layers.0.self_attn",
                    layer_type=mod.LayerType.FULL_ATTN,
                    block_size=4,
                    block_bytes=16,
                    layout=mod.TensorLayout.TOKEN_FIRST,
                    num_heads=1,
                    head_bytes=None,
                    token_first_layout=None,
                    token_stride_bytes=None,
                    head_stride_bytes=None,
                    live_head_bytes=None,
                    block_stride_bytes=16,
                    block_id_group_index=0,
                    head_segments=(),
                    physical_region_id="model.layers.0.self_attn",
                    region_base_offset_bytes=0),
                "model.layers.1.linear_attn.state0":
                mod.KVCacheRegion(
                    layer_name="model.layers.1.linear_attn.state0",
                    layer_type=mod.LayerType.MAMBA_STATE,
                    block_size=None,
                    block_bytes=8,
                    layout=mod.TensorLayout.BLOCKS_FIRST,
                    num_heads=1,
                    head_bytes=None,
                    token_first_layout=None,
                    token_stride_bytes=None,
                    head_stride_bytes=None,
                    live_head_bytes=None,
                    block_stride_bytes=8,
                    block_id_group_index=1,
                    head_segments=(),
                    physical_region_id="model.layers.1.linear_attn.state0",
                    region_base_offset_bytes=0),
            }
        },
        fa_block_ids=(0, ),
        mamba_block_ids=(0, ),
        block_ids_by_group=((0, ), (0, )),
        fa_num_tokens=4,
        mamba_num_tokens=4,
    )
    remote_metadata = (_remote_worker_metadata(
        dp_rank=0,
        tp_rank=0,
        worker_id="prefill-rank0",
        tcp_host="10.0.0.1",
        tcp_port=32100,
        transport="zmq",
        regions=(
            _registered_memory_region(
                region_id="model.layers.0.self_attn",
                nbytes=16,
                page_bytes=16,
            ),
            _registered_memory_region(
                region_id="model.layers.1.linear_attn.state0",
                nbytes=8,
                page_bytes=8,
            ),
        ),
    ), )

    updated = mod.TPUConnectorV2Worker.apply_remote_source_region_metadata(
        metadata, remote_metadata, dp_rank=0)

    assert updated is metadata
    missing_region_metadata = (_remote_worker_metadata(
        dp_rank=0,
        tp_rank=0,
        worker_id="prefill-rank0",
        tcp_host="10.0.0.1",
        tcp_port=32100,
        transport="zmq",
        regions=(_registered_memory_region(
            region_id="model.layers.0.self_attn",
            nbytes=16,
            page_bytes=16,
        ), ),
    ), )
    with pytest.raises(ValueError, match="linear_attn.state0"):
        mod.TPUConnectorV2Worker.apply_remote_source_region_metadata(
            metadata, missing_region_metadata, dp_rank=0)


def test_process_send_load_uses_strided_engine_for_decode_rank_ops(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingEngine:

        def __init__(self):
            self.registered = []

        def register_other_remote_metadata(self, metadata):
            self.registered.append(metadata)

    class RecordingBridge:

        def __init__(self):
            self.transfer_engine = RecordingEngine()
            self.calls = []
            self.sessions = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            self.calls.append(
                (remote_tp_rank, tuple(ops), write_session, dp_rank))
            return sum(op.segment_bytes * op.num_segments for op in ops)

        def new_destination_write_session(self):
            session = _RecordingWriteSession()
            self.sessions.append(session)
            return session

    remote_metadata = (
        _remote_worker_metadata(
            dp_rank=0,
            tp_rank=0,
            worker_id="prefill-rank0",
            tcp_host="10.0.0.1",
            tcp_port=32100,
            transport="zmq",
            regions=(_registered_memory_region(region_id="rank0.layer",
                                               nbytes=1024,
                                               page_bytes=1024), ),
        ),
        _remote_worker_metadata(
            dp_rank=0,
            tp_rank=2,
            worker_id="prefill-rank2",
            tcp_host="10.0.0.2",
            tcp_port=32102,
            transport="zmq",
            regions=(_registered_memory_region(region_id="rank2.layer",
                                               nbytes=1024,
                                               page_bytes=1024), ),
        ),
    )
    op_rank0 = mod.StridedSegmentOp(src_offset_bytes=0,
                                    dst_offset_bytes=4,
                                    segment_bytes=8,
                                    src_stride_bytes=16,
                                    dst_stride_bytes=32,
                                    num_segments=2,
                                    source_region_id="rank0.layer",
                                    destination_region_id="rank0.layer",
                                    layer_name="rank0.layer",
                                    layer_type=mod.LayerType.FULL_ATTN)
    op_rank2 = mod.StridedSegmentOp(src_offset_bytes=128,
                                    dst_offset_bytes=256,
                                    segment_bytes=4,
                                    src_stride_bytes=64,
                                    dst_stride_bytes=64,
                                    num_segments=3,
                                    source_region_id="rank2.layer",
                                    destination_region_id="rank2.layer",
                                    layer_name="rank2.layer",
                                    layer_type=mod.LayerType.FULL_ATTN)
    bridge = RecordingBridge()
    worker = mod.TPUConnectorV2Worker(object())
    worker.set_strided_transfer_bridge(bridge)
    _disable_v2_pull_start(worker)
    req_meta = types.SimpleNamespace(uuid=7,
                                     remote_block_ids=[1, 2],
                                     remote_metadata=remote_metadata,
                                     remote_rank_ops_by_decode_rank={
                                         0: {
                                             "2": (op_rank2, ),
                                             0: (op_rank0, ),
                                         }
                                     })
    connector_metadata = types.SimpleNamespace(
        reqs_to_send={}, reqs_to_load={"req-1": req_meta})

    copied = worker.process_send_load(connector_metadata)

    assert copied == 28
    assert bridge.transfer_engine.registered == [remote_metadata]
    assert [call[0] for call in bridge.calls] == [0, 2]
    assert len(bridge.sessions) == 1
    assert bridge.calls[0][2] is bridge.sessions[0]
    assert bridge.calls[1][2] is bridge.sessions[0]
    assert bridge.sessions[0].flushes == 1
    assert bridge.sessions[0].discards == 0
    assert bridge.calls[0][1][0] == mod.StridedSegmentOp(
        src_offset_bytes=0,
        dst_offset_bytes=4,
        segment_bytes=8,
        src_stride_bytes=16,
        dst_stride_bytes=32,
        num_segments=2,
        source_region_id="rank0.layer",
        destination_region_id="rank0.layer",
        layer_name="rank0.layer",
        layer_type=mod.LayerType.FULL_ATTN,
    )
    assert [{
        "source_region_id": op.source_region_id,
        "destination_region_id": op.destination_region_id,
        "src_offset_bytes": op.src_offset_bytes,
        "dst_offset_bytes": op.dst_offset_bytes,
        "segment_bytes": op.segment_bytes,
        "src_stride_bytes": op.src_stride_bytes,
        "dst_stride_bytes": op.dst_stride_bytes,
        "num_segments": op.num_segments,
        "layer_name": op.layer_name,
    } for op in bridge.calls[1][1]] == [{
        "source_region_id": "rank2.layer",
        "destination_region_id": "rank2.layer",
        "src_offset_bytes": 128,
        "dst_offset_bytes": 256,
        "segment_bytes": 4,
        "src_stride_bytes": 64,
        "dst_stride_bytes": 64,
        "num_segments": 3,
        "layer_name": "rank2.layer",
    }]
    assert all(call[2] is bridge.sessions[0] for call in bridge.calls)
    assert all(call[3] == 0 for call in bridge.calls)
    (completion, ) = _pop_worker_completions(mod, worker)
    assert completion == mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                            uuid=7,
                                                            dp_rank=0,
                                                            tp_rank=0,
                                                            success=True,
                                                            error="")


def test_worker_rejects_ambiguous_remote_dp_rank_before_pull(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingBridge:

        def __init__(self):
            self.calls = []

    worker = mod.TPUConnectorV2Worker(object())
    bridge = RecordingBridge()
    worker.set_strided_transfer_bridge(bridge)
    req_meta = types.SimpleNamespace(
        uuid=9,
        remote_block_ids=[1],
        remote_metadata=(
            _remote_worker_metadata(dp_rank=0,
                                    tp_rank=0,
                                    worker_id="p-dp0-tp0",
                                    tcp_host="10.0.0.1",
                                    tcp_port=32100,
                                    transport="zmq",
                                    regions=()),
            _remote_worker_metadata(dp_rank=1,
                                    tp_rank=0,
                                    worker_id="p-dp1-tp0",
                                    tcp_host="10.0.0.2",
                                    tcp_port=32101,
                                    transport="zmq",
                                    regions=()),
        ),
        remote_rank_ops_by_decode_rank={
            0: {
                0: (mod.StridedSegmentOp(src_offset_bytes=0,
                                         dst_offset_bytes=0,
                                         segment_bytes=1,
                                         src_stride_bytes=1,
                                         dst_stride_bytes=1,
                                         num_segments=1,
                                         source_region_id="layer.0",
                                         destination_region_id="layer.0",
                                         layer_name="layer.0",
                                         layer_type=mod.LayerType.FULL_ATTN), )
            }
        },
    )

    with pytest.raises(ValueError, match="remote_dp_rank"):
        worker._process_one_strided_load(
            "req-ambiguous",
            req_meta,
            req_meta.remote_rank_ops_by_decode_rank[0],
        )
    assert bridge.calls == []


def test_process_send_load_raises_without_strided_ops(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = mod.TPUConnectorV2Worker(object())
    connector_metadata = types.SimpleNamespace(
        reqs_to_send={},
        reqs_to_load={
            "req-1":
            types.SimpleNamespace(remote_metadata=(), remote_block_ids=[1])
        },
    )

    with pytest.raises(RuntimeError, match="remote_rank_ops_by_decode_rank"):
        worker.process_send_load(connector_metadata)


def test_process_send_load_does_not_fallback_for_v2_producer_send(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    worker = mod.TPUConnectorV2Worker(object())
    worker.tp_rank = 0
    worker._coord_send = {}
    worker._coord_lock = None
    connector_metadata = types.SimpleNamespace(
        reqs_to_send={
            "req-1": types.SimpleNamespace(uuid=42, expiration_time=123.0)
        },
        reqs_to_load={},
    )

    result = worker.process_send_load(connector_metadata)

    assert result == 0
    assert worker._coord_send[42].req_id == "req-1"
    assert worker._coord_send[42].slot_idx == -1


def test_v2_scheduler_carries_decode_rank_ops_into_load_metadata(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        parallel_config=types.SimpleNamespace(data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    remote_rank_ops_by_decode_rank = {
        0: {
            "1": [{
                "source_region_id": "layer.0",
                "destination_region_id": "layer.0",
                "src_offset_bytes": 0,
                "dst_offset_bytes": 8,
                "segment_bytes": 4,
                "src_stride_bytes": 16,
                "dst_stride_bytes": 16,
                "num_segments": 2,
                "layer_name": "layer.0",
                "layer_type": mod.LayerType.FULL_ATTN.value,
            }]
        }
    }
    request = types.SimpleNamespace(request_id="req-1",
                                    prompt_token_ids=list(range(9)),
                                    kv_transfer_params={
                                        "uuid": 123,
                                        "remote_block_ids": [[2]],
                                        "remote_host": "10.0.0.9",
                                        "remote_port": 7000,
                                        "remote_rank_ops_by_decode_rank":
                                        remote_rank_ops_by_decode_rank,
                                        "remote_dp_rank": 2,
                                    })

    scheduler.update_state_after_alloc(request, _blocks([7]), 8)

    load_meta = scheduler.reqs_to_load["req-1"]
    assert load_meta.remote_rank_ops_by_decode_rank == {
        0: {
            1: (mod.StridedSegmentOp(src_offset_bytes=0,
                                     dst_offset_bytes=8,
                                     segment_bytes=4,
                                     src_stride_bytes=16,
                                     dst_stride_bytes=16,
                                     num_segments=2,
                                     source_region_id="layer.0",
                                     destination_region_id="layer.0",
                                     layer_name="layer.0",
                                     layer_type=mod.LayerType.FULL_ATTN), )
        }
    }
    assert load_meta.remote_dp_rank == 2


def test_v2_worker_auto_installs_strided_transfer_engine(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    torch = pytest.importorskip("torch")

    class FakeEngine:

        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.regions = []
            FakeEngine.instances.append(self)

        def register_local_region(self, *, buffer, region):
            self.regions.append({
                "buffer": buffer,
                "region": region,
            })

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def local_metadata(self):
            return types.SimpleNamespace(
                to_dict=lambda: {
                    "dp_rank": self.kwargs["local_dp_rank"],
                    "tp_rank": self.kwargs["local_tp_rank"],
                    "worker_id": self.kwargs["local_worker_id"],
                    "tcp_host": self.kwargs["listen_host"],
                    "tcp_port": self.kwargs["listen_port"],
                    "transport": "zmq",
                    "regions": [],
                })

    monkeypatch.setattr(mod, "StridedKVTransferEngine", FakeEngine)

    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
        parallel_config=types.SimpleNamespace(data_parallel_rank=2,
                                              tensor_parallel_size=4))
    worker = mod.TPUConnectorV2Worker(config)
    host = _local_test_host()
    kv_transfer_port = _pick_strided_base_port()
    worker.tp_rank = 3
    worker.host_ip = host
    worker.kv_transfer_port = kv_transfer_port

    class MambaSpec:
        pass

    raw_mamba = torch.zeros(64, dtype=torch.int8)
    fa_cache = raw_mamba.view(torch.uint8).as_strided((2, 16), (32, 1), 0)
    conv_state = raw_mamba.view(torch.bfloat16).as_strided((2, 2), (16, 1), 0)
    ssm_state = raw_mamba.view(torch.float32).as_strided((2, 2), (8, 1), 4)
    mamba_cache = [conv_state, ssm_state]
    runner = types.SimpleNamespace(
        kv_caches=[fa_cache, mamba_cache],
        kv_cache_raw_tensors=[raw_mamba],
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=(
            types.SimpleNamespace(layer_names=("model.layers.0.self_attn", ),
                                  kv_cache_spec=object()),
            types.SimpleNamespace(layer_names=("model.layers.1.linear_attn", ),
                                  kv_cache_spec=MambaSpec()),
        )),
    )
    worker.named_kv_caches = {
        "model.layers.0.self_attn": fa_cache,
        "model.layers.1.linear_attn": mamba_cache,
    }

    worker.register_runner(runner)

    assert worker.strided_bridge is not None
    assert worker.strided_transfer_engine is FakeEngine.instances[0]
    assert worker.strided_transfer_engine.started
    assert worker.strided_transfer_engine.kwargs == {
        "local_dp_rank": 2,
        "local_tp_rank": 3,
        "local_worker_id": "dp2-tp3",
        "listen_host": host,
        "listen_port": kv_transfer_port + 10000 + 2 * 4 + 3,
        "transport": "zmq",
        "timeout_s": 120.0,
    }
    assert worker.strided_transfer_engine.regions == [
        {
            "buffer":
            raw_mamba,
            "region":
            mod.RegisteredMemoryRegion(
                region_id="__raw_kv_cache_0",
                nbytes=raw_mamba.numel() * raw_mamba.element_size(),
                page_bytes=32,
            ),
        },
    ]
    fa_region = worker._kv_cache_region_templates["model.layers.0.self_attn"]
    state0_region = worker._kv_cache_region_templates[
        "model.layers.1.linear_attn.state0"]
    state1_region = worker._kv_cache_region_templates[
        "model.layers.1.linear_attn.state1"]
    assert fa_region.physical_region_id == "__raw_kv_cache_0"
    assert fa_region.block_id_group_index == 0
    assert fa_region.region_base_offset_bytes == 0
    assert fa_region.block_stride_bytes == 32
    assert state0_region.physical_region_id == "__raw_kv_cache_0"
    assert state0_region.block_id_group_index == 1
    assert state0_region.region_base_offset_bytes == 0
    assert state0_region.block_stride_bytes == 32
    assert state1_region.physical_region_id == "__raw_kv_cache_0"
    assert state1_region.block_id_group_index == 1
    assert state1_region.region_base_offset_bytes == 16
    assert state1_region.block_stride_bytes == 32


def test_v2_decode_worker_installs_client_engine_without_starting_server(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    torch = pytest.importorskip("torch")

    class FakeEngine:

        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.regions = []
            FakeEngine.instances.append(self)

        def register_local_region(self, *, buffer, region):
            self.regions.append(region.region_id)

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def local_metadata(self):
            return types.SimpleNamespace(
                to_dict=lambda: {
                    "dp_rank": self.kwargs["local_dp_rank"],
                    "tp_rank": self.kwargs["local_tp_rank"],
                    "worker_id": self.kwargs["local_worker_id"],
                    "tcp_host": self.kwargs["listen_host"],
                    "tcp_port": self.kwargs["listen_port"],
                    "transport": "zmq",
                    "regions": [],
                })

    monkeypatch.setattr(mod, "StridedKVTransferEngine", FakeEngine)

    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        parallel_config=types.SimpleNamespace(data_parallel_rank=2,
                                              tensor_parallel_size=4))
    worker = mod.TPUConnectorV2Worker(config)
    host = _local_test_host()
    kv_transfer_port = _pick_strided_base_port()
    worker.tp_rank = 3
    worker.host_ip = host
    worker.kv_transfer_port = kv_transfer_port
    worker.group_is_mamba = (False, )
    runner = types.SimpleNamespace(
        kv_caches=[torch.zeros(6, dtype=torch.uint8)],
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=(types.SimpleNamespace(
                layer_names=("model.layers.0.self_attn", )), )),
    )
    worker.named_kv_caches = {"model.layers.0.self_attn": runner.kv_caches[0]}

    worker.register_runner(runner)

    assert worker.strided_bridge is not None
    assert worker.strided_transfer_engine is FakeEngine.instances[0]
    assert worker.strided_transfer_engine.started is False
    assert worker.strided_transfer_engine.kwargs["listen_host"] == host
    assert worker.strided_transfer_engine.kwargs[
        "listen_port"] == kv_transfer_port + 10000 + 2 * 4 + 3
    assert worker.strided_transfer_engine.kwargs["transport"] == "zmq"
    assert worker.strided_transfer_engine.regions == [
        "model.layers.0.self_attn"
    ]
    handshake = worker.get_handshake_metadata()
    assert handshake is not None
    assert handshake.remote_metadata.tp_rank == 3
    assert handshake.remote_metadata.regions[0].region_id == (
        "model.layers.0.self_attn")
    assert handshake.fa_group_indices == (0, )
    assert handshake.mamba_group_indices == ()


def _strided_port_probe_worker(
    mod,
    *,
    dp_rank,
    tp_rank,
    tp_size,
    pcp_rank,
    pcp_size,
    kv_transfer_port,
    host_ip,
):
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
        parallel_config=types.SimpleNamespace(
            data_parallel_rank=dp_rank,
            tensor_parallel_size=tp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )
    worker = mod.TPUConnectorV2Worker(config)
    worker.tp_rank = tp_rank
    worker.pcp_rank = pcp_rank
    worker.host_ip = host_ip
    worker.kv_transfer_port = kv_transfer_port
    return worker


def _strided_port_probe_rows(workers):
    return tuple({
        "dp_rank": worker._local_dp_rank(),
        "tp_rank": worker._local_tp_rank(),
        "pcp_rank": getattr(worker, "pcp_rank", None),
        "transfer_rank": worker._local_transfer_rank(),
        "worker_id": worker._local_worker_id(),
        "listen_port": worker._strided_listen_port(),
    } for worker in workers)


def test_v2_strided_listen_ports_are_unique_for_tp4_without_pcp(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    host = _local_test_host()
    kv_transfer_port = _pick_strided_base_port()
    first_listen_port = kv_transfer_port + 10000

    workers = [
        _strided_port_probe_worker(mod,
                                   dp_rank=0,
                                   tp_rank=tp_rank,
                                   tp_size=4,
                                   pcp_rank=0,
                                   pcp_size=1,
                                   kv_transfer_port=kv_transfer_port,
                                   host_ip=host) for tp_rank in range(4)
    ]
    rows = _strided_port_probe_rows(workers)

    assert [row["listen_port"] for row in rows
            ] == [first_listen_port + offset for offset in range(4)]
    assert [row["transfer_rank"] for row in rows] == [0, 1, 2, 3]
    assert len({row["listen_port"] for row in rows}) == len(rows)
    assert [row["worker_id"] for row in rows] == [
        "dp0-tp0",
        "dp0-tp1",
        "dp0-tp2",
        "dp0-tp3",
    ]


def test_v2_strided_listen_ports_are_unique_for_pcp4_tp1(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    host = _local_test_host()
    kv_transfer_port = _pick_strided_base_port()
    first_listen_port = kv_transfer_port + 10000

    workers = [
        _strided_port_probe_worker(mod,
                                   dp_rank=0,
                                   tp_rank=0,
                                   tp_size=1,
                                   pcp_rank=pcp_rank,
                                   pcp_size=4,
                                   kv_transfer_port=kv_transfer_port,
                                   host_ip=host) for pcp_rank in range(4)
    ]
    rows = _strided_port_probe_rows(workers)

    detail = json.dumps(rows, indent=2, sort_keys=True)
    assert [row["listen_port"] for row in rows
            ] == [first_listen_port + offset for offset in range(4)], detail
    assert [row["transfer_rank"] for row in rows] == [0, 1, 2, 3], detail
    assert len({row["listen_port"] for row in rows}) == len(rows), detail
    assert len({row["transfer_rank"] for row in rows}) == len(rows), detail
    assert len({row["worker_id"] for row in rows}) == len(rows), detail
    assert [row["worker_id"] for row in rows] == [
        "dp0-pcp0-tp0",
        "dp0-pcp1-tp0",
        "dp0-pcp2-tp0",
        "dp0-pcp3-tp0",
    ], detail


def _single_layer_source_metadata(mod):
    return mod.ConnectorMetadataV2(
        req_id=11,
        block_size=4,
        kv_source_layout=mod.KVParallelLayout(full_attn_pcp_size=1,
                                              full_attn_tp_size=1,
                                              linear_attn_pcp_size=1,
                                              linear_attn_tp_size=1),
        kv_caches={
            0: {
                "layer.0":
                mod.KVCacheRegion(layer_name="layer.0",
                                  layer_type=mod.LayerType.FULL_ATTN,
                                  block_size=4,
                                  block_bytes=16,
                                  layout=mod.TensorLayout.TOKEN_FIRST,
                                  num_heads=1,
                                  head_bytes=None,
                                  token_first_layout=None,
                                  token_stride_bytes=None,
                                  head_stride_bytes=None,
                                  live_head_bytes=None,
                                  block_stride_bytes=16,
                                  block_id_group_index=0,
                                  head_segments=(),
                                  physical_region_id="layer.0",
                                  region_base_offset_bytes=0),
            }
        },
        fa_block_ids=(2, ),
        mamba_block_ids=(),
        block_ids_by_group=((2, ), ),
        fa_num_tokens=4,
        mamba_num_tokens=None,
    )


def _single_layer_destination(mod):
    return mod.LocalDecodeAllocation(
        rank=0,
        block_size=4,
        kv_caches={
            "layer.0":
            mod.KVCacheRegion(layer_name="layer.0",
                              layer_type=mod.LayerType.FULL_ATTN,
                              block_size=4,
                              block_bytes=16,
                              layout=mod.TensorLayout.TOKEN_FIRST,
                              num_heads=1,
                              head_bytes=None,
                              token_first_layout=None,
                              token_stride_bytes=None,
                              head_stride_bytes=None,
                              live_head_bytes=None,
                              block_stride_bytes=16,
                              block_id_group_index=0,
                              head_segments=(),
                              physical_region_id="layer.0",
                              region_base_offset_bytes=0),
        },
        fa_block_ids=(7, ),
        mamba_block_ids=(),
        block_ids_by_group=((7, ), ),
        fa_num_tokens=4,
        mamba_num_tokens=None,
    )


def _single_layer_topology(mod):
    return mod.TpKVTopology(
        local_layout=mod.KVParallelLayout(full_attn_pcp_size=1,
                                          full_attn_tp_size=1,
                                          linear_attn_pcp_size=1,
                                          linear_attn_tp_size=1),
        block_size=4,
        tp_rank=0,
        total_num_kv_heads=1,
        total_num_mamba_key_heads=0,
        total_num_mamba_heads=0,
    )


def _single_layer_handshake(mod, *, tp_rank=0):
    remote = _remote_worker_metadata(
        dp_rank=0,
        tp_rank=tp_rank,
        worker_id=f"p{tp_rank}",
        tcp_host=f"10.0.0.{tp_rank + 1}",
        tcp_port=32100 + tp_rank,
        transport="zmq",
        regions=(_registered_memory_region(region_id="layer.0",
                                           nbytes=64,
                                           page_bytes=16), ),
    )
    source = _single_layer_source_metadata(mod)
    return mod.TPUConnectorV2HandshakeMetadata(
        remote_metadata=remote,
        kv_source_layout=source.kv_source_layout,
        kv_caches=source.kv_caches[0],
        topology=_single_layer_topology(mod),
        fa_group_indices=(0, ),
        mamba_group_indices=(),
    )


def test_v2_wire_metadata_json_round_trips_real_kv_transfer_params(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    metadata = _single_layer_source_metadata(mod)
    destination = _single_layer_destination(mod)
    topology = _single_layer_topology(mod)
    planner = mod.ContiguousHeadTPTransferPlanner()
    pull_meta = planner.build_pull_meta(metadata, topology)
    plans = planner.lower(metadata, topology, destination, pull_meta)

    assert not hasattr(mod.TPUConnectorV2Scheduler,
                       "serialize_strided_wire_metadata")
    assert not hasattr(mod.TPUConnectorV2Scheduler,
                       "deserialize_strided_wire_metadata")
    assert not hasattr(mod.TPUConnectorV2Scheduler,
                       "install_default_strided_remote_metadata")
    assert not hasattr(mod.TPUConnectorV2Worker, "build_pull_meta")
    assert not hasattr(mod.TPUConnectorV2Worker, "lower_transfer_plan")
    assert not hasattr(mod.TPUConnectorV2Worker, "pull_rank_transfer_plans")
    assert not hasattr(mod.TPUConnectorV2StridedBridge, "pull_rank_plan")
    assert not hasattr(mod.TPUConnectorV2StridedBridge, "pull_rank_plans")
    assert not hasattr(mod.HeadMapping, "to_dict")
    assert not hasattr(mod.HeadMapping, "from_mapping")
    assert not hasattr(mod.SourceBlockRef, "to_dict")
    assert not hasattr(mod.SourceBlockRef, "from_mapping")
    assert not hasattr(mod.PullMeta, "to_dict")
    assert not hasattr(mod.RankTransferPlan, "to_dict")
    assert not hasattr(mod.RankTransferPlan, "from_mapping")

    kv_transfer_params = {
        "strided_source_metadata": metadata.to_dict(),
        "remote_rank_ops_by_decode_rank": {
            0: {
                rank: [op.to_dict() for op in plan.ops]
                for rank, plan in plans.items()
            }
        },
    }
    region_wire = kv_transfer_params["strided_source_metadata"]["kv_caches"][
        0]["layer.0"]
    assert not {"rank", "base_addr", "block_id_index"} & region_wire.keys()
    decoded = json.loads(json.dumps(kv_transfer_params))

    source_metadata = mod.ConnectorMetadataV2.from_mapping(
        decoded["strided_source_metadata"])
    rank_ops_by_decode_rank = (
        mod.TPUConnectorV2Worker.remote_rank_ops_by_decode_rank_from_wire(
            decoded["remote_rank_ops_by_decode_rank"]))

    assert source_metadata == metadata
    assert rank_ops_by_decode_rank[0][0] == plans[0].ops


def test_v2_runtime_setters_reject_mapping_shims(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    source_metadata = _single_layer_source_metadata(mod)
    remote_metadata = _remote_worker_metadata(
        dp_rank=0,
        tp_rank=0,
        worker_id="p0",
        tcp_host="10.0.0.1",
        tcp_port=32100,
        transport="zmq",
        regions=(_registered_memory_region(region_id="layer.0",
                                           nbytes=64,
                                           page_bytes=16), ),
    )

    with pytest.raises(TypeError, match="ConnectorMetadataV2"):
        scheduler.set_strided_source_metadata(source_metadata.to_dict())
    with pytest.raises(TypeError, match="RemoteWorkerMetadata"):
        scheduler.set_remote_metadata([remote_metadata.to_dict()])

    scheduler.set_strided_source_metadata(source_metadata)
    scheduler.set_remote_metadata([remote_metadata])


def test_v2_worker_decode_rank_ops_use_single_op_sequence_shape(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    op = mod.StridedSegmentOp(src_offset_bytes=0,
                              dst_offset_bytes=0,
                              segment_bytes=1,
                              src_stride_bytes=1,
                              dst_stride_bytes=1,
                              num_segments=1,
                              source_region_id="layer.0",
                              destination_region_id="layer.0",
                              layer_name="layer.0",
                              layer_type=mod.LayerType.FULL_ATTN)

    assert mod.TPUConnectorV2Worker._remote_rank_ops_by_decode_rank_from_req_meta(
        types.SimpleNamespace(remote_rank_ops_by_decode_rank={0: {
            1: (op, )
        }})) == {
            0: {
                1: (op, )
            }
        }

    with pytest.raises(TypeError, match="StridedSegmentOp"):
        mod.TPUConnectorV2Worker._remote_rank_ops_by_decode_rank_from_req_meta(
            types.SimpleNamespace(
                remote_rank_ops_by_decode_rank={0: {
                    1: [op.to_dict()]
                }}))
    with pytest.raises(TypeError, match="op sequence"):
        mod.TPUConnectorV2Worker._remote_rank_ops_by_decode_rank_from_req_meta(
            types.SimpleNamespace(
                remote_rank_ops_by_decode_rank={0: {
                    1: {
                        "ops": [op]
                    }
                }}))
    assert mod.TPUConnectorV2Worker._remote_rank_ops_by_decode_rank_from_req_meta(
        types.SimpleNamespace(ops=[op], remote_tp_rank=0)) == {}


def test_v2_worker_remote_rank_ops_by_decode_rank_rejects_wire_dicts(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    op = mod.StridedSegmentOp(src_offset_bytes=0,
                              dst_offset_bytes=0,
                              segment_bytes=1,
                              src_stride_bytes=1,
                              dst_stride_bytes=1,
                              num_segments=1,
                              source_region_id="layer.0",
                              destination_region_id="layer.0",
                              layer_name="layer.0",
                              layer_type=mod.LayerType.FULL_ATTN)

    by_decode_rank = (
        mod.TPUConnectorV2Worker._remote_rank_ops_by_decode_rank_from_req_meta(
            types.SimpleNamespace(
                remote_rank_ops_by_decode_rank={1: {
                    0: (op, )
                }})))
    assert by_decode_rank == {1: {0: (op, )}}

    with pytest.raises(TypeError, match="StridedSegmentOp"):
        mod.TPUConnectorV2Worker._remote_rank_ops_by_decode_rank_from_req_meta(
            types.SimpleNamespace(
                remote_rank_ops_by_decode_rank={1: {
                    0: [op.to_dict()]
                }}))


def _two_tp_source_metadata(mod):
    layout = mod.KVParallelLayout(full_attn_pcp_size=1,
                                  full_attn_tp_size=2,
                                  linear_attn_pcp_size=1,
                                  linear_attn_tp_size=2)
    return mod.ConnectorMetadataV2(
        req_id=11,
        block_size=4,
        kv_source_layout=layout,
        kv_caches={
            rank: {
                "layer.0":
                mod.KVCacheRegion(layer_name="layer.0",
                                  layer_type=mod.LayerType.FULL_ATTN,
                                  block_size=4,
                                  block_bytes=16,
                                  layout=mod.TensorLayout.TOKEN_FIRST,
                                  num_heads=1,
                                  head_bytes=None,
                                  token_first_layout=None,
                                  token_stride_bytes=None,
                                  head_stride_bytes=None,
                                  live_head_bytes=None,
                                  block_stride_bytes=16,
                                  block_id_group_index=0,
                                  head_segments=(),
                                  physical_region_id="layer.0",
                                  region_base_offset_bytes=0),
            }
            for rank in (0, 1)
        },
        fa_block_ids=(2, ),
        mamba_block_ids=(),
        block_ids_by_group=((2, ), ),
        fa_num_tokens=4,
        mamba_num_tokens=None,
    )


def _two_tp_decode_handshake(mod, tp_rank):
    source = _two_tp_source_metadata(mod)
    layout = source.kv_source_layout
    return mod.TPUConnectorV2HandshakeMetadata(
        remote_metadata=_remote_worker_metadata(
            dp_rank=0,
            tp_rank=tp_rank,
            worker_id=f"d{tp_rank}",
            tcp_host=f"10.0.1.{tp_rank + 1}",
            tcp_port=42100 + tp_rank,
            transport="zmq",
            regions=(_registered_memory_region(region_id="layer.0",
                                               nbytes=64,
                                               page_bytes=16), ),
        ),
        kv_source_layout=layout,
        kv_caches={
            "layer.0":
            mod.KVCacheRegion(layer_name="layer.0",
                              layer_type=mod.LayerType.FULL_ATTN,
                              block_size=4,
                              block_bytes=16,
                              layout=mod.TensorLayout.TOKEN_FIRST,
                              num_heads=1,
                              head_bytes=None,
                              token_first_layout=None,
                              token_stride_bytes=None,
                              head_stride_bytes=None,
                              live_head_bytes=None,
                              block_stride_bytes=16,
                              block_id_group_index=0,
                              head_segments=(),
                              physical_region_id="layer.0",
                              region_base_offset_bytes=0),
        },
        topology=mod.TpKVTopology(local_layout=layout,
                                  block_size=4,
                                  tp_rank=tp_rank,
                                  total_num_kv_heads=2,
                                  total_num_mamba_key_heads=0,
                                  total_num_mamba_heads=0),
        fa_group_indices=(0, ),
        mamba_group_indices=(),
    )


def test_v2_scheduler_generates_decode_rank_plans_for_multi_tp_handshake(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=2,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({
        0: _two_tp_decode_handshake(mod, 0),
        1: _two_tp_decode_handshake(mod, 1),
    })
    source_metadata = _two_tp_source_metadata(mod)
    request = types.SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(5)),
        kv_transfer_params={
            "uuid":
            99,
            "remote_block_ids": [[2]],
            "remote_host":
            "10.0.0.9",
            "remote_port":
            7000,
            "remote_metadata": [
                _remote_worker_metadata(dp_rank=0,
                                        tp_rank=0,
                                        worker_id="p0",
                                        tcp_host="10.0.0.1",
                                        tcp_port=32100,
                                        transport="zmq",
                                        regions=(_registered_memory_region(
                                            region_id="layer.0",
                                            nbytes=64,
                                            page_bytes=16), )).to_dict(),
                _remote_worker_metadata(dp_rank=0,
                                        tp_rank=1,
                                        worker_id="p1",
                                        tcp_host="10.0.0.2",
                                        tcp_port=32101,
                                        transport="zmq",
                                        regions=(_registered_memory_region(
                                            region_id="layer.0",
                                            nbytes=64,
                                            page_bytes=16), )).to_dict(),
            ],
            "remote_dp_rank":
            0,
            "strided_source_metadata":
            source_metadata.to_dict(),
        },
    )

    scheduler.update_state_after_alloc(request, _blocks([7]), 4)

    load_meta = scheduler.reqs_to_load["req-1"]
    assert not hasattr(load_meta, "remote_rank_ops")
    assert set(load_meta.remote_rank_ops_by_decode_rank) == {0, 1}
    assert tuple(load_meta.remote_rank_ops_by_decode_rank[0]) == (0, )
    assert tuple(load_meta.remote_rank_ops_by_decode_rank[1]) == (1, )
    assert load_meta.remote_rank_ops_by_decode_rank[0][0][
        0].dst_offset_bytes == 112
    assert load_meta.remote_rank_ops_by_decode_rank[1][1][
        0].dst_offset_bytes == 112


def test_v2_worker_uses_only_local_decode_rank_ops(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingBridge:

        def __init__(self):
            self.transfer_engine = types.SimpleNamespace(
                register_other_remote_metadata=lambda metadata: None)
            self.calls = []
            self.sessions = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            self.calls.append(
                (remote_tp_rank, tuple(ops), write_session, dp_rank))
            return 1

        def new_destination_write_session(self):
            session = _RecordingWriteSession()
            self.sessions.append(session)
            return session

    op_rank0 = mod.StridedSegmentOp(src_offset_bytes=0,
                                    dst_offset_bytes=0,
                                    segment_bytes=1,
                                    src_stride_bytes=1,
                                    dst_stride_bytes=1,
                                    num_segments=1,
                                    source_region_id="layer.0",
                                    destination_region_id="layer.0",
                                    layer_name="layer.0",
                                    layer_type=mod.LayerType.FULL_ATTN)
    op_rank1 = mod.StridedSegmentOp(src_offset_bytes=16,
                                    dst_offset_bytes=32,
                                    segment_bytes=2,
                                    src_stride_bytes=4,
                                    dst_stride_bytes=4,
                                    num_segments=1,
                                    source_region_id="layer.0",
                                    destination_region_id="layer.0",
                                    layer_name="layer.0",
                                    layer_type=mod.LayerType.FULL_ATTN)
    bridge = RecordingBridge()
    worker = mod.TPUConnectorV2Worker(object())
    worker.tp_rank = 1
    worker.set_strided_transfer_bridge(bridge)
    _disable_v2_pull_start(worker)
    req_meta = types.SimpleNamespace(
        uuid=123,
        remote_block_ids=[1],
        remote_metadata=(
            _remote_worker_metadata(dp_rank=0,
                                    tp_rank=0,
                                    worker_id="p0",
                                    tcp_host="10.0.0.1",
                                    tcp_port=32100,
                                    transport="zmq",
                                    regions=()),
            _remote_worker_metadata(dp_rank=0,
                                    tp_rank=1,
                                    worker_id="p1",
                                    tcp_host="10.0.0.2",
                                    tcp_port=32101,
                                    transport="zmq",
                                    regions=()),
        ),
        remote_rank_ops_by_decode_rank={
            0: {
                0: (op_rank0, ),
            },
            1: {
                1: (op_rank1, ),
            },
        })

    copied = worker.process_send_load(
        types.SimpleNamespace(reqs_to_send={},
                              reqs_to_load={"req-1": req_meta}))

    assert copied == 1
    assert len(bridge.sessions) == 1
    assert bridge.calls == [(1, (op_rank1, ), bridge.sessions[0], 0)]
    assert bridge.sessions[0].flushes == 1
    assert bridge.sessions[0].discards == 0
    (completion, ) = _pop_worker_completions(mod, worker)
    assert completion == mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                            uuid=123,
                                                            dp_rank=0,
                                                            tp_rank=1,
                                                            success=True,
                                                            error="")


def test_v2_scheduler_attaches_source_metadata_on_producer_finish(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    source_metadata = _single_layer_source_metadata(mod)
    scheduler.set_strided_source_metadata(source_metadata)
    scheduler.set_remote_metadata([
        _remote_worker_metadata(
            dp_rank=0,
            tp_rank=0,
            worker_id="p0",
            tcp_host="10.0.0.1",
            tcp_port=32100,
            transport="zmq",
            regions=(_registered_memory_region(region_id="layer.0",
                                               nbytes=64,
                                               page_bytes=16), ),
        )
    ])
    request = types.SimpleNamespace(request_id="req-1",
                                    status=_finished_length_capped_status())

    delay, params = scheduler.request_finished_all_groups(request, ([2], ))

    assert delay is True
    assert params["v2_ack_host"] == "127.0.0.1"
    assert params["v2_ack_port"] == 9600
    assert params["strided_source_metadata"] == source_metadata.to_dict()
    assert params["remote_metadata"][0]["regions"][0] == {
        "region_id": "layer.0",
        "nbytes": 64,
        "page_bytes": 16,
    }


def test_v2_scheduler_builds_source_metadata_from_worker_handshake_on_producer_finish(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({0: _single_layer_handshake(mod)})
    request = types.SimpleNamespace(request_id="req-1",
                                    status=_finished_length_capped_status())

    delay, params = scheduler.request_finished_all_groups(request, ([2], ))

    assert delay is True
    source_metadata = mod.ConnectorMetadataV2.from_mapping(
        params["strided_source_metadata"])
    assert source_metadata.kv_caches[0]["layer.0"].layer_name == "layer.0"
    assert source_metadata.fa_block_ids == (2, )
    assert source_metadata.mamba_block_ids == ()
    assert params["remote_dp_rank"] == 0
    assert params["v2_ack_host"] == "127.0.0.1"
    assert params["v2_ack_port"] == 9600
    assert params["remote_metadata"][0]["tcp_port"] == 32100


def test_v2_scheduler_generates_remote_rank_ops_after_decode_alloc(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        parallel_config=types.SimpleNamespace(data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_strided_decode_metadata(
        topology=_single_layer_topology(mod),
        destination=_single_layer_destination(mod),
    )
    request = types.SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(5)),
        kv_transfer_params={
            "uuid":
            99,
            "remote_block_ids": [[2]],
            "remote_host":
            "10.0.0.9",
            "remote_port":
            7000,
            "remote_metadata": [{
                "dp_rank":
                0,
                "tp_rank":
                0,
                "worker_id":
                "p0",
                "tcp_host":
                "10.0.0.1",
                "tcp_port":
                32100,
                "transport":
                "zmq",
                "regions": [{
                    "region_id": "layer.0",
                    "nbytes": 64,
                    "page_bytes": 16,
                }],
            }],
            "strided_source_metadata":
            _single_layer_source_metadata(mod).to_dict(),
        })

    scheduler.update_state_after_alloc(request, _blocks([7]), 4)

    load_meta = scheduler.reqs_to_load["req-1"]
    assert load_meta.remote_metadata[0].dp_rank == 0
    assert load_meta.remote_metadata[0].tp_rank == 0
    assert load_meta.remote_metadata[0].regions[0].region_id == "layer.0"
    assert not hasattr(load_meta.remote_metadata[0].regions[0], "base_addr")
    assert not hasattr(load_meta, "remote_rank_ops")
    assert tuple(load_meta.remote_rank_ops_by_decode_rank) == (0, )
    rank_ops = load_meta.remote_rank_ops_by_decode_rank[0]
    assert tuple(rank_ops) == (0, )
    op = rank_ops[0][0]
    assert op.layer_name == "layer.0"
    assert op.source_region_id == "layer.0"
    assert op.destination_region_id == "layer.0"
    assert "source_base_addr" not in op.to_dict()
    assert "destination_base_addr" not in op.to_dict()


def test_v2_scheduler_generates_remote_rank_ops_from_worker_handshake_after_decode_alloc(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({0: _single_layer_handshake(mod)})
    source_metadata = _single_layer_source_metadata(mod)
    request = types.SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(5)),
        kv_transfer_params={
            "uuid":
            99,
            "remote_block_ids": [[2]],
            "remote_host":
            "10.0.0.9",
            "remote_port":
            7000,
            "remote_metadata":
            [_single_layer_handshake(mod).remote_metadata.to_dict()],
            "remote_dp_rank":
            0,
            "strided_source_metadata":
            source_metadata.to_dict(),
        },
    )

    scheduler.update_state_after_alloc(request, _blocks([7]), 4)

    load_meta = scheduler.reqs_to_load["req-1"]
    assert not hasattr(load_meta, "remote_rank_ops")
    assert tuple(load_meta.remote_rank_ops_by_decode_rank) == (0, )
    rank_ops = load_meta.remote_rank_ops_by_decode_rank[0]
    assert tuple(rank_ops) == (0, )
    op = rank_ops[0][0]
    assert op.source_region_id == "layer.0"
    assert op.destination_region_id == "layer.0"
    assert op.src_offset_bytes == 32
    assert op.dst_offset_bytes == 112


def test_v2_scheduler_keeps_single_nonzero_decode_rank_ops_by_decode_rank(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        parallel_config=types.SimpleNamespace(data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    op = mod.StridedSegmentOp(src_offset_bytes=0,
                              dst_offset_bytes=0,
                              segment_bytes=1,
                              src_stride_bytes=1,
                              dst_stride_bytes=1,
                              num_segments=1,
                              source_region_id="layer.0",
                              destination_region_id="layer.0",
                              layer_name="layer.0",
                              layer_type=mod.LayerType.FULL_ATTN)
    scheduler._lower_decode_rank_ops = lambda req_meta, source: {
        1: {
            0: (op, )
        }
    }
    req_meta = types.SimpleNamespace(
        uuid=99,
        remote_block_ids=[[2]],
        local_block_ids=[[7]],
        remote_metadata=(_single_layer_handshake(mod).remote_metadata, ),
        remote_dp_rank=0)

    scheduler._maybe_generate_strided_rank_ops(
        req_meta,
        {"strided_source_metadata": _single_layer_source_metadata(mod)},
        num_external_tokens=4,
    )

    assert not hasattr(req_meta, "remote_rank_ops")
    assert req_meta.remote_rank_ops_by_decode_rank == {1: {0: (op, )}}


def test_v2_scheduler_skips_noop_load_with_stale_source_metadata(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({0: _single_layer_handshake(mod)})
    request = types.SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(5)),
        kv_transfer_params={
            "uuid":
            99,
            "remote_block_ids": [[2]],
            "remote_host":
            "10.0.0.9",
            "remote_port":
            7000,
            "remote_metadata":
            [_single_layer_handshake(mod).remote_metadata.to_dict()],
            "remote_dp_rank":
            0,
            "strided_source_metadata":
            _single_layer_source_metadata(mod).to_dict(),
        },
    )

    scheduler.update_state_after_alloc(request, object(), 0)

    assert scheduler.reqs_to_load == {}


def test_v2_scheduler_does_not_redispatch_finished_uuid(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({0: _single_layer_handshake(mod)})
    request = types.SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(5)),
        kv_transfer_params={
            "uuid":
            99,
            "remote_block_ids": [[2]],
            "remote_host":
            "10.0.0.9",
            "remote_port":
            7000,
            "remote_metadata":
            [_single_layer_handshake(mod).remote_metadata.to_dict()],
            "remote_dp_rank":
            0,
            "strided_source_metadata":
            _single_layer_source_metadata(mod).to_dict(),
        },
    )

    scheduler.update_state_after_alloc(request, _blocks([7]), 4)
    first_meta = scheduler.build_connector_meta()
    assert set(first_meta.reqs_to_load) == {"req-1"}

    scheduler.update_connector_output(
        types.SimpleNamespace(finished_recving={"req-1"}))
    scheduler.update_state_after_alloc(request, object(), 0)
    second_meta = scheduler.build_connector_meta()

    assert second_meta.reqs_to_load == {}


def test_v2_scheduler_requires_local_block_ids_for_real_load(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({0: _single_layer_handshake(mod)})
    req_meta = types.SimpleNamespace(
        uuid=99,
        local_block_ids=None,
        remote_block_ids=[[2]],
        remote_metadata=(_single_layer_handshake(mod).remote_metadata, ),
        remote_dp_rank=0,
    )

    with pytest.raises(RuntimeError, match="local_block_ids"):
        scheduler._maybe_generate_strided_rank_ops(
            req_meta,
            {"strided_source_metadata": _single_layer_source_metadata(mod)},
            num_external_tokens=4,
        )


def test_v2_scheduler_uses_external_token_count_for_partial_fa_page(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({0: _single_layer_handshake(mod)})
    source_metadata = mod.ConnectorMetadataV2(
        req_id=11,
        block_size=4,
        kv_source_layout=mod.KVParallelLayout(full_attn_pcp_size=1,
                                              full_attn_tp_size=1,
                                              linear_attn_pcp_size=1,
                                              linear_attn_tp_size=1),
        kv_caches=_single_layer_source_metadata(mod).kv_caches,
        fa_block_ids=(2, 3),
        mamba_block_ids=(),
        block_ids_by_group=((2, 3), ),
        fa_num_tokens=8,
        mamba_num_tokens=None,
    )
    request = types.SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(4)),
        kv_transfer_params={
            "uuid":
            99,
            "remote_block_ids": [[2, 3]],
            "remote_host":
            "10.0.0.9",
            "remote_port":
            7000,
            "remote_metadata":
            [_single_layer_handshake(mod).remote_metadata.to_dict()],
            "remote_dp_rank":
            0,
            "strided_source_metadata":
            source_metadata.to_dict(),
        },
    )

    scheduler.update_state_after_alloc(request, _blocks([7]), 3)

    load_meta = scheduler.reqs_to_load["req-1"]
    assert load_meta.strided_source_metadata.fa_num_tokens == 3
    assert not hasattr(load_meta, "remote_rank_ops")
    assert tuple(load_meta.remote_rank_ops_by_decode_rank) == (0, )
    rank_ops = load_meta.remote_rank_ops_by_decode_rank[0]
    assert tuple(rank_ops) == (0, )
    op = rank_ops[0][0]
    assert op.src_offset_bytes == 32
    assert op.dst_offset_bytes == 112
    assert op.segment_bytes == 4
    assert op.src_stride_bytes == 4
    assert op.dst_stride_bytes == 4
    assert op.num_segments == 3


def test_v2_scheduler_offsets_fa_transfer_after_local_prefix_hit(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({0: _single_layer_handshake(mod)})
    source_metadata = mod.ConnectorMetadataV2(
        req_id=11,
        block_size=4,
        kv_source_layout=mod.KVParallelLayout(full_attn_pcp_size=1,
                                              full_attn_tp_size=1,
                                              linear_attn_pcp_size=1,
                                              linear_attn_tp_size=1),
        kv_caches=_single_layer_source_metadata(mod).kv_caches,
        fa_block_ids=(100, 101, 102, 103, 104),
        mamba_block_ids=(),
        block_ids_by_group=((100, 101, 102, 103, 104), ),
        fa_num_tokens=20,
        mamba_num_tokens=None,
    )
    request = types.SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(21)),
        kv_transfer_params={
            "uuid":
            99,
            "remote_block_ids": [[100, 101, 102, 103, 104]],
            "remote_host":
            "10.0.0.9",
            "remote_port":
            7000,
            "remote_metadata":
            [_single_layer_handshake(mod).remote_metadata.to_dict()],
            "remote_dp_rank":
            0,
            "strided_source_metadata":
            source_metadata.to_dict(),
        },
    )

    scheduler.update_state_after_alloc(request,
                                       _blocks([200, 201, 202, 203, 204]), 12)

    load_meta = scheduler.reqs_to_load["req-1"]
    rank_ops = load_meta.remote_rank_ops_by_decode_rank[0]
    assert tuple(rank_ops) == (0, )
    assert [(op.src_offset_bytes, op.dst_offset_bytes, op.num_segments)
            for op in rank_ops[0]] == [
                (102 * 16, 202 * 16, 4),  # source block 102 -> dst block 202
                (103 * 16, 203 * 16, 4),  # source block 103 -> dst block 203
                (104 * 16, 204 * 16, 4),  # source block 104 -> dst block 204
            ]


def test_v2_scheduler_clamps_external_tokens_to_source_fa_blocks(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({0: _single_layer_handshake(mod)})
    request = types.SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(8)),
        kv_transfer_params={
            "uuid":
            99,
            "remote_block_ids": [[2]],
            "remote_host":
            "10.0.0.9",
            "remote_port":
            7000,
            "remote_metadata":
            [_single_layer_handshake(mod).remote_metadata.to_dict()],
            "remote_dp_rank":
            0,
            "strided_source_metadata":
            _single_layer_source_metadata(mod).to_dict(),
        },
    )

    scheduler.update_state_after_alloc(request, _blocks([7]), 7)

    load_meta = scheduler.reqs_to_load["req-1"]
    assert load_meta.strided_source_metadata.fa_num_tokens == 4
    rank_ops = load_meta.remote_rank_ops_by_decode_rank[0]
    assert tuple(rank_ops) == (0, )
    op = rank_ops[0][0]
    assert op.num_segments == 4


def test_v2_scheduler_raises_when_decode_load_lacks_source_metadata(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    scheduler = mod.TPUConnectorV2Scheduler(config)
    scheduler.set_xfer_handshake_metadata({0: _single_layer_handshake(mod)})
    request = types.SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(5)),
        kv_transfer_params={
            "uuid":
            99,
            "remote_block_ids": [[2]],
            "remote_host":
            "10.0.0.9",
            "remote_port":
            7000,
            "remote_metadata":
            [_single_layer_handshake(mod).remote_metadata.to_dict()],
            "remote_dp_rank":
            0,
        },
    )

    with pytest.raises(RuntimeError, match="strided_source_metadata"):
        scheduler.update_state_after_alloc(request, _blocks([7]), 4)


def test_v2_connector_forwards_xfer_handshake_metadata(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
        cache_config=types.SimpleNamespace(block_size=4),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1,
                                              data_parallel_rank=0),
    )
    worker_connector = mod.TPUConnectorV2(config, mod.KVConnectorRole.WORKER)
    handshake = _single_layer_handshake(mod)
    worker_connector.connector_worker._handshake_metadata = handshake

    assert worker_connector.get_handshake_metadata() is handshake

    scheduler_connector = mod.TPUConnectorV2(config,
                                             mod.KVConnectorRole.SCHEDULER)
    scheduler_connector.set_xfer_handshake_metadata({0: handshake})

    scheduler = scheduler_connector.connector_scheduler
    assert scheduler.remote_metadata == (handshake.remote_metadata, )
    assert scheduler._handshake_metadata_by_tp_rank == {0: handshake}


def test_v2_scheduler_ends_after_worker_completion_meta(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    req_meta = types.SimpleNamespace(uuid=456,
                                     remote_host="10.0.0.9",
                                     remote_port=7000,
                                     v2_ack_host="10.0.0.9",
                                     v2_ack_port=7600,
                                     remote_rank_ops_by_decode_rank={
                                         0: {},
                                         1: {},
                                     })
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))
    scheduler._mark_recv_lifecycle_planned(req_id="req-1",
                                           uuid=456,
                                           req_meta=req_meta,
                                           expected_tp_ranks={0, 1})
    scheduler.ends = []
    scheduler._send_v2_end = lambda meta: scheduler.ends.append(
        (meta.v2_ack_host, meta.v2_ack_port, meta.uuid))
    completion0 = mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                     uuid=456,
                                                     dp_rank=0,
                                                     tp_rank=0,
                                                     success=True,
                                                     error="")
    completion1 = mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                     uuid=456,
                                                     dp_rank=0,
                                                     tp_rank=1,
                                                     success=True,
                                                     error="")
    worker_meta = mod.TPUConnectorV2WorkerMeta(
        completions=(completion0, )).aggregate(
            mod.TPUConnectorV2WorkerMeta(completions=(completion1, )))
    output = types.SimpleNamespace(finished_recving=None,
                                   kv_connector_worker_meta=worker_meta)

    scheduler.update_connector_output(output)

    assert output.finished_recving == {"req-1"}
    assert scheduler.ends == [("10.0.0.9", 7600, 456)]


def test_v2_scheduler_rejects_multihost_scalar_ack_port(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    dist_utils = sys.modules["vllm_torchtpu.distributed.utils"]
    monkeypatch.setattr(dist_utils, "get_node_id", lambda: 1)
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))
    req_meta = types.SimpleNamespace(uuid=456,
                                     v2_ack_host=("10.0.0.10", "10.0.0.11"),
                                     v2_ack_port=9600)

    with pytest.raises(ValueError, match="per-host ACK port"):
        scheduler._resolve_v2_side_channel_endpoint(req_meta)


def test_v2_scheduler_attaches_per_node_ack_ports_for_multihost(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=True),
            parallel_config=types.SimpleNamespace(data_parallel_rank=2),
        ))
    kv_transfer_params = {
        "remote_host": ("10.0.0.10", "10.0.0.11"),
    }

    scheduler._attach_v2_side_channel_endpoint(kv_transfer_params)

    assert kv_transfer_params["v2_ack_host"] == ("10.0.0.10", "10.0.0.11")
    assert kv_transfer_params["v2_ack_port"] == [9602, 9603]


def test_v2_scheduler_raises_on_failed_worker_completion(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    req_meta = types.SimpleNamespace(uuid=456,
                                     remote_host="10.0.0.9",
                                     remote_port=7000,
                                     v2_ack_host="10.0.0.9",
                                     v2_ack_port=7600,
                                     remote_rank_ops_by_decode_rank={0: {}})
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))
    scheduler._mark_recv_lifecycle_planned(req_id="req-1",
                                           uuid=456,
                                           req_meta=req_meta,
                                           expected_tp_ranks={0})
    scheduler.ends = []
    scheduler._send_v2_end = lambda meta: scheduler.ends.append(meta.uuid)
    output = types.SimpleNamespace(
        finished_recving=None,
        kv_connector_worker_meta=mod.TPUConnectorV2WorkerMeta(completions=(
            mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                               uuid=456,
                                               dp_rank=0,
                                               tp_rank=0,
                                               success=False,
                                               error="tcp failed"), )))

    with pytest.raises(RuntimeError, match="tcp failed"):
        scheduler.update_connector_output(output)

    assert output.finished_recving is None
    assert scheduler.ends == []


def test_v2_scheduler_processes_success_completion_in_failed_batch(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)
    req_meta_failed = types.SimpleNamespace(
        uuid=456,
        remote_host="10.0.0.9",
        remote_port=7000,
        v2_ack_host="10.0.0.9",
        v2_ack_port=7600,
        remote_rank_ops_by_decode_rank={0: {}},
    )
    req_meta_ok = types.SimpleNamespace(
        uuid=457,
        remote_host="10.0.0.10",
        remote_port=7000,
        v2_ack_host="10.0.0.10",
        v2_ack_port=7600,
        remote_rank_ops_by_decode_rank={0: {}},
    )
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))
    scheduler._mark_recv_lifecycle_planned(req_id="req-failed",
                                           uuid=456,
                                           req_meta=req_meta_failed,
                                           expected_tp_ranks={0})
    scheduler._mark_recv_lifecycle_planned(req_id="req-ok",
                                           uuid=457,
                                           req_meta=req_meta_ok,
                                           expected_tp_ranks={0})
    scheduler.ends = []
    scheduler._send_v2_end = lambda meta: scheduler.ends.append(meta.uuid)
    output = types.SimpleNamespace(
        finished_recving=None,
        kv_connector_worker_meta=mod.TPUConnectorV2WorkerMeta(completions=(
            mod.TPUConnectorV2WorkerCompletion(req_id="req-failed",
                                               uuid=456,
                                               dp_rank=0,
                                               tp_rank=0,
                                               success=False,
                                               error="tcp failed"),
            mod.TPUConnectorV2WorkerCompletion(req_id="req-ok",
                                               uuid=457,
                                               dp_rank=0,
                                               tp_rank=0,
                                               success=True,
                                               error=""),
        )))

    with pytest.raises(RuntimeError, match="tcp failed"):
        scheduler.update_connector_output(output)

    assert output.finished_recving == {"req-ok"}
    assert scheduler.ends == [457]


def test_v2_scheduler_raises_when_pull_end_rejected(monkeypatch):
    mod = _load_v2_module(monkeypatch)
    req_meta = types.SimpleNamespace(uuid=456,
                                     remote_host="10.0.0.9",
                                     remote_port=7000,
                                     v2_ack_host="10.0.0.9",
                                     v2_ack_port=7600,
                                     remote_rank_ops_by_decode_rank={0: {}})
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))
    scheduler._mark_recv_lifecycle_planned(req_id="req-1",
                                           uuid=456,
                                           req_meta=req_meta,
                                           expected_tp_ranks={0})

    def reject_end(meta):
        raise RuntimeError("producer rejected strided KV pull end: "
                           "expired uuid=456")

    scheduler._send_v2_end = reject_end
    output = types.SimpleNamespace(
        finished_recving=None,
        kv_connector_worker_meta=mod.TPUConnectorV2WorkerMeta(
            completions=(mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                            uuid=456,
                                                            dp_rank=0,
                                                            tp_rank=0,
                                                            success=True,
                                                            error=""), )))

    with pytest.raises(RuntimeError, match="expired uuid=456"):
        scheduler.update_connector_output(output)

    assert output.finished_recving is None


def test_v2_worker_single_decode_rank_reports_completion_meta(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingBridge:

        def __init__(self):
            self.transfer_engine = types.SimpleNamespace(
                register_other_remote_metadata=lambda metadata: None)
            self.sessions = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            return 1

        def new_destination_write_session(self):
            session = _RecordingWriteSession()
            self.sessions.append(session)
            return session

    req_meta = types.SimpleNamespace(
        uuid=123,
        remote_block_ids=[1],
        remote_host="10.0.0.9",
        remote_port=7000,
        remote_metadata=(_remote_worker_metadata(dp_rank=0,
                                                 tp_rank=0,
                                                 worker_id="p0",
                                                 tcp_host="10.0.0.1",
                                                 tcp_port=32100,
                                                 transport="zmq",
                                                 regions=()), ),
        remote_rank_ops_by_decode_rank={
            0: {
                0: (mod.StridedSegmentOp(src_offset_bytes=0,
                                         dst_offset_bytes=0,
                                         segment_bytes=1,
                                         src_stride_bytes=1,
                                         dst_stride_bytes=1,
                                         num_segments=1,
                                         source_region_id="layer.0",
                                         destination_region_id="layer.0",
                                         layer_name="layer.0",
                                         layer_type=mod.LayerType.FULL_ATTN), )
            }
        })
    metadata = types.SimpleNamespace(reqs_to_send={},
                                     reqs_to_load={"req-1": req_meta})

    rank0 = mod.TPUConnectorV2Worker(object())
    rank0.tp_rank = 0
    rank0.set_strided_transfer_bridge(RecordingBridge())
    _disable_v2_pull_start(rank0)
    rank0.process_send_load(metadata)

    (completion, ) = _pop_worker_completions(mod, rank0)
    assert completion == mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                            uuid=123,
                                                            dp_rank=0,
                                                            tp_rank=0,
                                                            success=True,
                                                            error="")


def test_v2_worker_reports_only_local_decode_rank_completion(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingBridge:

        def __init__(self):
            self.transfer_engine = types.SimpleNamespace(
                register_other_remote_metadata=lambda metadata: None)
            self.calls = []
            self.sessions = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            self.calls.append(
                (remote_tp_rank, tuple(ops), write_session, dp_rank))
            return 1

        def new_destination_write_session(self):
            session = _RecordingWriteSession()
            self.sessions.append(session)
            return session

    op_rank0 = mod.StridedSegmentOp(src_offset_bytes=0,
                                    dst_offset_bytes=0,
                                    segment_bytes=1,
                                    src_stride_bytes=1,
                                    dst_stride_bytes=1,
                                    num_segments=1,
                                    source_region_id="layer.0",
                                    destination_region_id="layer.0",
                                    layer_name="layer.0",
                                    layer_type=mod.LayerType.FULL_ATTN)
    op_rank1 = mod.StridedSegmentOp(src_offset_bytes=16,
                                    dst_offset_bytes=16,
                                    segment_bytes=1,
                                    src_stride_bytes=1,
                                    dst_stride_bytes=1,
                                    num_segments=1,
                                    source_region_id="layer.0",
                                    destination_region_id="layer.0",
                                    layer_name="layer.0",
                                    layer_type=mod.LayerType.FULL_ATTN)
    req_meta = types.SimpleNamespace(
        uuid=456,
        remote_block_ids=[1],
        remote_host="10.0.0.9",
        remote_port=7000,
        v2_ack_host="10.0.0.9",
        v2_ack_port=7600,
        remote_dp_rank=0,
        remote_metadata=(),
        remote_rank_ops_by_decode_rank={
            0: {
                0: (op_rank0, ),
            },
            1: {
                1: (op_rank1, ),
            },
        },
    )
    metadata = types.SimpleNamespace(reqs_to_send={},
                                     reqs_to_load={"req-1": req_meta})

    rank0_bridge = RecordingBridge()
    rank0 = mod.TPUConnectorV2Worker(object())
    rank0.tp_rank = 0
    rank0.tp_size = 2
    rank0.set_strided_transfer_bridge(rank0_bridge)
    _disable_v2_pull_start(rank0)

    rank0.process_send_load(metadata)

    assert rank0_bridge.calls == [(0, (op_rank0, ), rank0_bridge.sessions[0],
                                   0)]
    (completion, ) = _pop_worker_completions(mod, rank0)
    assert completion == mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                            uuid=456,
                                                            dp_rank=0,
                                                            tp_rank=0,
                                                            success=True,
                                                            error="")


def test_v2_scheduler_ends_after_all_decode_tp_ranks_finish(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingBridge:

        def __init__(self):
            self.transfer_engine = types.SimpleNamespace(
                register_other_remote_metadata=lambda metadata: None)
            self.calls = []
            self.sessions = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            self.calls.append(
                (remote_tp_rank, tuple(ops), write_session, dp_rank))
            return 1

        def new_destination_write_session(self):
            session = _RecordingWriteSession()
            self.sessions.append(session)
            return session

    op_rank0 = mod.StridedSegmentOp(src_offset_bytes=0,
                                    dst_offset_bytes=0,
                                    segment_bytes=1,
                                    src_stride_bytes=1,
                                    dst_stride_bytes=1,
                                    num_segments=1,
                                    source_region_id="layer.0",
                                    destination_region_id="layer.0",
                                    layer_name="layer.0",
                                    layer_type=mod.LayerType.FULL_ATTN)
    op_rank1 = mod.StridedSegmentOp(src_offset_bytes=16,
                                    dst_offset_bytes=16,
                                    segment_bytes=1,
                                    src_stride_bytes=1,
                                    dst_stride_bytes=1,
                                    num_segments=1,
                                    source_region_id="layer.0",
                                    destination_region_id="layer.0",
                                    layer_name="layer.0",
                                    layer_type=mod.LayerType.FULL_ATTN)
    req_meta = types.SimpleNamespace(
        uuid=456,
        remote_block_ids=[1],
        remote_host="10.0.0.9",
        remote_port=7000,
        v2_ack_host="10.0.0.9",
        v2_ack_port=7600,
        remote_dp_rank=0,
        remote_metadata=(),
        remote_rank_ops_by_decode_rank={
            0: {
                0: (op_rank0, ),
            },
            1: {
                1: (op_rank1, ),
            },
        },
    )
    metadata = types.SimpleNamespace(reqs_to_send={},
                                     reqs_to_load={"req-1": req_meta})

    rank0_bridge = RecordingBridge()
    rank0 = mod.TPUConnectorV2Worker(object())
    rank0.tp_rank = 0
    rank0.tp_size = 2
    rank0.set_strided_transfer_bridge(rank0_bridge)
    _disable_v2_pull_start(rank0)

    rank0.process_send_load(metadata)
    (completion0, ) = _pop_worker_completions(mod, rank0)
    assert rank0_bridge.calls == [(0, (op_rank0, ), rank0_bridge.sessions[0],
                                   0)]

    rank1_bridge = RecordingBridge()
    rank1 = mod.TPUConnectorV2Worker(object())
    rank1.tp_rank = 1
    rank1.tp_size = 2
    rank1.set_strided_transfer_bridge(rank1_bridge)
    _disable_v2_pull_start(rank1)

    rank1.process_send_load(metadata)
    (completion1, ) = _pop_worker_completions(mod, rank1)
    assert rank1_bridge.calls == [(1, (op_rank1, ), rank1_bridge.sessions[0],
                                   0)]

    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))
    scheduler._mark_recv_lifecycle_planned(req_id="req-1",
                                           uuid=456,
                                           req_meta=req_meta,
                                           expected_tp_ranks={0, 1})
    scheduler.ends = []
    scheduler._send_v2_end = lambda meta: scheduler.ends.append(
        (meta.v2_ack_host, meta.v2_ack_port, meta.uuid))

    output = types.SimpleNamespace(
        finished_recving=None,
        kv_connector_worker_meta=mod.TPUConnectorV2WorkerMeta(
            completions=(completion0, )))
    scheduler.update_connector_output(output)
    assert output.finished_recving is None
    assert scheduler.ends == []

    output.kv_connector_worker_meta = mod.TPUConnectorV2WorkerMeta(
        completions=(completion1, ))
    scheduler.update_connector_output(output)
    assert output.finished_recving == {"req-1"}
    assert scheduler.ends == [("10.0.0.9", 7600, 456)]


def test_v2_worker_get_finished_does_not_report_strided_recv(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    rank0 = mod.TPUConnectorV2Worker(object())
    rank0.tp_rank = 0
    rank0.tp_size = 2
    rank0._coord_lock = mod.threading.Lock()
    rank0._coord_send = {}
    rank0._coord_recv = {}
    rank0._coord_done_sending = set()
    rank0._coord_done_recving = set()
    req_meta = types.SimpleNamespace(
        uuid=458,
        remote_host="10.0.0.9",
        remote_port=7000,
        remote_rank_ops_by_decode_rank={
            0: {},
            1: {},
        },
    )

    rank0._finish_strided_recv(req_id="req-3",
                               req_meta=req_meta,
                               success=True,
                               error="")
    assert rank0._coord_get_finished() == (set(), set())

    (completion, ) = _pop_worker_completions(mod, rank0)
    assert completion == mod.TPUConnectorV2WorkerCompletion(req_id="req-3",
                                                            uuid=458,
                                                            dp_rank=0,
                                                            tp_rank=0,
                                                            success=True,
                                                            error="")
    assert rank0._coord_get_finished() == (set(), set())


def test_v2_worker_failed_completion_requires_error_message(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    rank0 = mod.TPUConnectorV2Worker(object())
    rank0.tp_rank = 0
    rank0.tp_size = 1
    rank0._coord_lock = mod.threading.Lock()
    rank0._coord_send = {}
    rank0._coord_recv = {}
    rank0._coord_done_sending = set()
    rank0._coord_done_recving = set()
    req_meta = types.SimpleNamespace(
        uuid=459,
        remote_rank_ops_by_decode_rank={0: {}},
    )

    with pytest.raises(TypeError):
        rank0._finish_strided_recv(req_id="req-4",
                                   req_meta=req_meta,
                                   success=False)


def test_v2_scheduler_waits_for_all_decode_tp_completions(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    req_meta = types.SimpleNamespace(
        uuid=457,
        remote_block_ids=[1],
        remote_host="10.0.0.9",
        remote_port=7000,
        v2_ack_host="10.0.0.9",
        v2_ack_port=7600,
        remote_dp_rank=0,
        remote_metadata=(),
        remote_rank_ops_by_decode_rank={
            0: {},
            1: {},
        },
    )
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))
    scheduler._mark_recv_lifecycle_planned(req_id="req-2",
                                           uuid=457,
                                           req_meta=req_meta,
                                           expected_tp_ranks={0, 1})
    scheduler.ends = []
    scheduler._send_v2_end = lambda meta: scheduler.ends.append(
        (meta.v2_ack_host, meta.v2_ack_port, meta.uuid))
    output = types.SimpleNamespace(
        finished_recving=None,
        kv_connector_worker_meta=mod.TPUConnectorV2WorkerMeta(
            completions=(mod.TPUConnectorV2WorkerCompletion(req_id="req-2",
                                                            uuid=457,
                                                            dp_rank=0,
                                                            tp_rank=1,
                                                            success=True,
                                                            error=""), )))

    scheduler.update_connector_output(output)

    assert output.finished_recving is None
    assert scheduler.ends == []

    output.kv_connector_worker_meta = mod.TPUConnectorV2WorkerMeta(
        completions=(mod.TPUConnectorV2WorkerCompletion(req_id="req-2",
                                                        uuid=457,
                                                        dp_rank=0,
                                                        tp_rank=0,
                                                        success=True,
                                                        error=""), ))
    scheduler.update_connector_output(output)

    assert output.finished_recving == {"req-2"}
    assert scheduler.ends == [("10.0.0.9", 7600, 457)]


def test_v2_scheduler_prunes_recv_lifecycle_after_end_ack(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    req_meta = types.SimpleNamespace(
        uuid=458,
        remote_block_ids=[1],
        v2_ack_host="10.0.0.9",
        v2_ack_port=7600,
        remote_rank_ops_by_decode_rank={0: {}},
    )
    scheduler = mod.TPUConnectorV2Scheduler(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0),
        ))
    scheduler._send_v2_end = lambda meta: None
    scheduler._mark_recv_lifecycle_planned(req_id="req-clean",
                                           uuid=458,
                                           req_meta=req_meta,
                                           expected_tp_ranks={0})
    scheduler._mark_recv_lifecycle_dispatched(req_id="req-clean", uuid=458)
    output = types.SimpleNamespace(
        finished_recving=None,
        kv_connector_worker_meta=mod.TPUConnectorV2WorkerMeta(
            completions=(mod.TPUConnectorV2WorkerCompletion(req_id="req-clean",
                                                            uuid=458,
                                                            dp_rank=0,
                                                            tp_rank=0,
                                                            success=True,
                                                            error=""), )))

    scheduler.update_connector_output(output)

    assert output.finished_recving == {"req-clean"}
    assert 458 not in scheduler._recv_lifecycle_by_uuid
    assert "req-clean" not in scheduler._recv_uuid_by_req_id


def test_v2_worker_prunes_strided_recv_lifecycle_after_request_leaves_metadata(
        monkeypatch):
    mod = _load_v2_module(monkeypatch)

    worker = mod.TPUConnectorV2Worker(
        types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(is_kv_producer=False),
            parallel_config=types.SimpleNamespace(data_parallel_rank=0,
                                                  tensor_parallel_size=1),
        ))
    req_meta = types.SimpleNamespace(uuid=459,
                                     remote_rank_ops_by_decode_rank={0: {}})

    assert worker._try_mark_strided_recv_started(459)
    worker._finish_strided_recv(req_id="req-clean",
                                req_meta=req_meta,
                                success=True,
                                error="")
    (completion, ) = _pop_worker_completions(mod, worker)

    assert completion.uuid == 459
    worker.process_send_load(
        types.SimpleNamespace(reqs_to_send={}, reqs_to_load={}))
    assert 459 not in worker._strided_recv_started_uuids
    assert 459 not in worker._strided_recv_done_uuids


def test_v2_worker_noop_load_does_not_report_done_recving(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    worker = mod.TPUConnectorV2Worker(object())
    worker.tp_rank = 0
    req_meta = types.SimpleNamespace(uuid=123,
                                     remote_block_ids=None,
                                     remote_host="10.0.0.9",
                                     remote_port=7000)

    copied = worker.process_send_load(
        types.SimpleNamespace(reqs_to_send={},
                              reqs_to_load={"req-1": req_meta}))

    assert copied == 0
    assert worker.build_connector_worker_meta() is None


def test_v2_worker_deduplicates_repeated_real_load_uuid(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingBridge:

        def __init__(self):
            self.transfer_engine = types.SimpleNamespace(
                register_other_remote_metadata=lambda metadata: None)
            self.calls = []
            self.sessions = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            self.calls.append(
                (remote_tp_rank, tuple(ops), write_session, dp_rank))
            return 1

        def new_destination_write_session(self):
            session = _RecordingWriteSession()
            self.sessions.append(session)
            return session

    bridge = RecordingBridge()
    worker = mod.TPUConnectorV2Worker(object())
    worker.tp_rank = 0
    worker.set_strided_transfer_bridge(bridge)
    _disable_v2_pull_start(worker)
    req_meta = types.SimpleNamespace(
        uuid=123,
        remote_block_ids=[1],
        remote_host="10.0.0.9",
        remote_port=7000,
        remote_metadata=(_remote_worker_metadata(dp_rank=0,
                                                 tp_rank=0,
                                                 worker_id="p0",
                                                 tcp_host="10.0.0.1",
                                                 tcp_port=32100,
                                                 transport="zmq",
                                                 regions=()), ),
        remote_rank_ops_by_decode_rank={
            0: {
                0: (mod.StridedSegmentOp(src_offset_bytes=0,
                                         dst_offset_bytes=0,
                                         segment_bytes=1,
                                         src_stride_bytes=1,
                                         dst_stride_bytes=1,
                                         num_segments=1,
                                         source_region_id="layer.0",
                                         destination_region_id="layer.0",
                                         layer_name="layer.0",
                                         layer_type=mod.LayerType.FULL_ATTN), )
            }
        })
    metadata = types.SimpleNamespace(reqs_to_send={},
                                     reqs_to_load={"req-1": req_meta})

    assert worker.process_send_load(metadata) == 1
    assert len(bridge.calls) == 1
    assert bridge.calls[0][2] is bridge.sessions[0]
    assert bridge.sessions[0].flushes == 1
    assert bridge.sessions[0].discards == 0
    (completion, ) = _pop_worker_completions(mod, worker)
    assert completion == mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                            uuid=123,
                                                            dp_rank=0,
                                                            tp_rank=0,
                                                            success=True,
                                                            error="")

    assert worker.process_send_load(metadata) == 0

    assert len(bridge.calls) == 1
    assert worker.build_connector_worker_meta() is None


def test_v2_worker_strided_failure_raises_without_completion(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class FailingBridge:

        def __init__(self):
            self.transfer_engine = types.SimpleNamespace(
                register_other_remote_metadata=lambda metadata: None)
            self.sessions = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            raise RuntimeError("tcp failed")

        def new_destination_write_session(self):
            session = _RecordingWriteSession()
            self.sessions.append(session)
            return session

    worker = mod.TPUConnectorV2Worker(object())
    worker.tp_rank = 0
    bridge = FailingBridge()
    worker.set_strided_transfer_bridge(bridge)
    _disable_v2_pull_start(worker)
    req_meta = types.SimpleNamespace(
        uuid=124,
        remote_block_ids=[1],
        remote_host="10.0.0.9",
        remote_port=7000,
        remote_metadata=(_remote_worker_metadata(dp_rank=0,
                                                 tp_rank=0,
                                                 worker_id="p0",
                                                 tcp_host="10.0.0.1",
                                                 tcp_port=32100,
                                                 transport="zmq",
                                                 regions=()), ),
        remote_rank_ops_by_decode_rank={
            0: {
                0: (mod.StridedSegmentOp(src_offset_bytes=0,
                                         dst_offset_bytes=0,
                                         segment_bytes=1,
                                         src_stride_bytes=1,
                                         dst_stride_bytes=1,
                                         num_segments=1,
                                         source_region_id="layer.0",
                                         destination_region_id="layer.0",
                                         layer_name="layer.0",
                                         layer_type=mod.LayerType.FULL_ATTN), )
            }
        })

    with pytest.raises(RuntimeError, match="tcp failed"):
        worker.process_send_load(
            types.SimpleNamespace(reqs_to_send={},
                                  reqs_to_load={"req-1": req_meta}))

    (completion, ) = _pop_worker_completions(mod, worker)
    assert completion == mod.TPUConnectorV2WorkerCompletion(req_id="req-1",
                                                            uuid=124,
                                                            dp_rank=0,
                                                            tp_rank=0,
                                                            success=False,
                                                            error="tcp failed")
    assert bridge.sessions[0].flushes == 0
    assert bridge.sessions[0].discards == 1


def test_v2_worker_pull_start_rejection_fails_before_copy(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingBridge:

        def __init__(self):
            self.transfer_engine = types.SimpleNamespace(
                register_other_remote_metadata=lambda metadata: None)
            self.calls = []
            self.sessions = []

        def pull_rank_ops_into_session(self, *, remote_tp_rank, ops,
                                       write_session, dp_rank):
            self.calls.append(
                (remote_tp_rank, tuple(ops), write_session, dp_rank))
            return 1

        def new_destination_write_session(self):
            session = _RecordingWriteSession()
            self.sessions.append(session)
            return session

    worker = mod.TPUConnectorV2Worker(object())
    worker.tp_rank = 0
    bridge = RecordingBridge()
    worker.set_strided_transfer_bridge(bridge)
    monkeypatch.setattr(worker,
                        "_request_v2_pull_start",
                        lambda req_meta:
                        (_ for _ in
                         ()).throw(RuntimeError("producer expired uuid=126")),
                        raising=False)
    req_meta = types.SimpleNamespace(
        uuid=126,
        remote_block_ids=[1],
        remote_host="10.0.0.9",
        remote_port=7000,
        v2_ack_host="10.0.0.9",
        v2_ack_port=7600,
        remote_dp_rank=0,
        remote_metadata=(_remote_worker_metadata(dp_rank=0,
                                                 tp_rank=0,
                                                 worker_id="p0",
                                                 tcp_host="10.0.0.1",
                                                 tcp_port=32100,
                                                 transport="zmq",
                                                 regions=()), ),
        remote_rank_ops_by_decode_rank={
            0: {
                0: (mod.StridedSegmentOp(src_offset_bytes=0,
                                         dst_offset_bytes=0,
                                         segment_bytes=1,
                                         src_stride_bytes=1,
                                         dst_stride_bytes=1,
                                         num_segments=1,
                                         source_region_id="layer.0",
                                         destination_region_id="layer.0",
                                         layer_name="layer.0",
                                         layer_type=mod.LayerType.FULL_ATTN), )
            }
        })

    with pytest.raises(RuntimeError, match="producer expired uuid=126"):
        worker.process_send_load(
            types.SimpleNamespace(reqs_to_send={},
                                  reqs_to_load={"req-1": req_meta}))

    (completion, ) = _pop_worker_completions(mod, worker)
    assert completion == mod.TPUConnectorV2WorkerCompletion(
        req_id="req-1",
        uuid=126,
        dp_rank=0,
        tp_rank=0,
        success=False,
        error="producer expired uuid=126",
    )
    assert bridge.calls == []
    assert bridge.sessions == []


def test_v2_worker_schedules_strided_pulls_on_coord_executor(monkeypatch):
    mod = _load_v2_module(monkeypatch)

    class RecordingExecutor:

        def __init__(self):
            self.calls = []

        def submit(self, fn, *args, **kwargs):
            self.calls.append((fn.__name__, args, kwargs))
            return types.SimpleNamespace(done=lambda: False)

    class RecordingBridge:

        def __init__(self):
            self.transfer_engine = types.SimpleNamespace(
                register_other_remote_metadata=lambda metadata: None)

    worker = mod.TPUConnectorV2Worker(object())
    worker.set_strided_transfer_bridge(RecordingBridge())
    worker._coord_executor = RecordingExecutor()
    req_meta = types.SimpleNamespace(
        uuid=125,
        remote_block_ids=[1],
        remote_metadata=(),
        remote_rank_ops_by_decode_rank={
            0: {
                0: (mod.StridedSegmentOp(src_offset_bytes=0,
                                         dst_offset_bytes=0,
                                         segment_bytes=1,
                                         src_stride_bytes=1,
                                         dst_stride_bytes=1,
                                         num_segments=1,
                                         source_region_id="layer.0",
                                         destination_region_id="layer.0",
                                         layer_name="layer.0",
                                         layer_type=mod.LayerType.FULL_ATTN), )
            }
        })

    copied = worker.process_send_load(
        types.SimpleNamespace(reqs_to_send={},
                              reqs_to_load={"req-1": req_meta}))

    assert copied == 0
    assert worker._coord_executor.calls[0][0] == (
        "_process_one_strided_load_fail_forward")
