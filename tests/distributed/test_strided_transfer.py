# SPDX-License-Identifier: Apache-2.0
"""Tests for byte-op based strided KV/state transfer."""

from __future__ import annotations

import logging
import socket
import sys
import threading
import types

import pytest

vllm = types.ModuleType("vllm")
vllm_logger = types.ModuleType("vllm.logger")
vllm_logger._VllmLogger = logging.Logger
vllm_logger.init_logger = logging.getLogger
sys.modules.setdefault("vllm", vllm)
sys.modules.setdefault("vllm.logger", vllm_logger)

import vllm_torchtpu.distributed.kv_transfer.v2.strided_transfer as strided_transfer  # noqa: E402

RemoteWorkerMetadata = getattr(strided_transfer, "RemoteWorkerMetadata", None)
RegisteredMemoryRegion = getattr(strided_transfer, "RegisteredMemoryRegion",
                                 None)
StridedKVTransferEngine = getattr(strided_transfer, "StridedKVTransferEngine",
                                  None)
StridedTransferOp = getattr(strided_transfer, "StridedTransferOp", None)


def test_public_engine_api_exists() -> None:
    assert RemoteWorkerMetadata is not None
    assert RegisteredMemoryRegion is not None
    assert StridedKVTransferEngine is not None
    assert StridedTransferOp is not None
    assert not hasattr(strided_transfer, "scatter_strided_ops")
    assert not hasattr(strided_transfer, "apply_strided_ops")
    assert not hasattr(strided_transfer, "gather_strided_ops")
    assert not hasattr(strided_transfer, "TensorSequenceByteBuffer")


def test_remote_worker_metadata_requires_explicit_regions() -> None:
    _require_api()

    with pytest.raises(TypeError):
        RemoteWorkerMetadata(dp_rank=0,
                             tp_rank=0,
                             worker_id="p0",
                             tcp_host="127.0.0.1",
                             tcp_port=9000,
                             transport="zmq")


def _require_api() -> None:
    if (RemoteWorkerMetadata is None or RegisteredMemoryRegion is None
            or StridedKVTransferEngine is None or StridedTransferOp is None):
        pytest.skip("new strided transfer engine API is covered by "
                    "test_public_engine_api_exists")


def _pattern(rank: int, offset: int) -> int:
    return (rank * 37 + offset * 11) & 0xFF


def _buffer(rank: int, size: int) -> bytearray:
    return bytearray(_pattern(rank, offset) for offset in range(size))


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _buffer_nbytes(buffer) -> int:
    nbytes = getattr(buffer, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return len(memoryview(buffer).cast("B"))


def _register_region(engine, buffer, *, region_id: str, page_bytes: int):
    return engine.register_local_region(
        buffer=buffer,
        region=RegisteredMemoryRegion(region_id=region_id,
                                      nbytes=_buffer_nbytes(buffer),
                                      page_bytes=page_bytes),
    )


def _engine() -> StridedKVTransferEngine:
    return StridedKVTransferEngine(local_dp_rank=0,
                                   local_tp_rank=0,
                                   listen_host="127.0.0.1",
                                   listen_port=0,
                                   transport="zmq")


def _assert_segments_match(src: bytes | bytearray, dst: bytes | bytearray,
                           ops: list[StridedTransferOp]) -> None:
    for op in ops:
        for i in range(op.num_segments):
            src_offset = op.src_offset_bytes + i * op.src_stride_bytes
            dst_offset = op.dst_offset_bytes + i * op.dst_stride_bytes
            assert dst[dst_offset:dst_offset +
                       op.segment_bytes] == src[src_offset:src_offset +
                                                op.segment_bytes]


def _apply_expected_strided_ops(src: bytes | bytearray, dst: bytearray,
                                ops: list[StridedTransferOp]) -> int:
    copied = 0
    for op in ops:
        for i in range(op.num_segments):
            src_offset = op.src_offset_bytes + i * op.src_stride_bytes
            dst_offset = op.dst_offset_bytes + i * op.dst_stride_bytes
            dst[dst_offset:dst_offset +
                op.segment_bytes] = src[src_offset:src_offset +
                                        op.segment_bytes]
            copied += op.segment_bytes
    return copied


def _make_zmq_producer(*,
                       dp_rank: int,
                       tp_rank: int,
                       source: bytes | bytearray,
                       worker_id: str | None = None):
    producer = StridedKVTransferEngine(local_dp_rank=dp_rank,
                                       local_tp_rank=tp_rank,
                                       local_worker_id=worker_id,
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    _register_region(producer,
                     source,
                     region_id="source",
                     page_bytes=max(1, _buffer_nbytes(source)))
    producer.start()
    return producer


def test_module_no_longer_exposes_plan_generation_api() -> None:
    removed_names = {
        "AxisRange",
        "TpSlice",
        "StridedTransferPlan",
        "StridedTransferPlanner",
        "PackedKvLayout",
        "ConvStateLayout",
        "GdnStateLayout",
        "build_tp_slices",
        "packed_kv_page_bytes",
        "packed_kv_useful_page_bytes",
        "conv_state_page_bytes",
        "gdn_state_page_bytes",
        "assign_compact_offsets",
        "split_plan_by_remote_rank",
        "plan_to_dict",
        "plan_from_dict",
        "gather_to_compact",
        "scatter_from_compact",
        "TensorSequenceByteBuffer",
    }
    for name in removed_names:
        assert not hasattr(strided_transfer, name)


def test_engine_requires_explicit_zmq_endpoint_contract() -> None:
    _require_api()
    with pytest.raises(TypeError):
        StridedKVTransferEngine(local_dp_rank=0, local_tp_rank=0)
    with pytest.raises(TypeError):
        StridedKVTransferEngine(local_dp_rank=0,
                                local_tp_rank=0,
                                listen_host="127.0.0.1",
                                listen_port=0)


def test_engine_exposes_raw_region_registration_in_metadata() -> None:
    _require_api()
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=12345,
                                     transport="zmq")
    region = _register_region(engine,
                              bytearray(8),
                              region_id="raw.layer",
                              page_bytes=4)

    assert region == RegisteredMemoryRegion(nbytes=8,
                                            region_id="raw.layer",
                                            page_bytes=4)
    assert engine.local_metadata().to_dict()["regions"] == [{
        "region_id": "raw.layer",
        "nbytes": 8,
        "page_bytes": 4,
    }]


def test_region_metadata_and_ops_use_region_relative_offsets() -> None:
    _require_api()
    producer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=0,
                                       listen_host="127.0.0.1",
                                       listen_port=12345,
                                       transport="zmq")
    consumer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=1,
                                       listen_host="127.0.0.1",
                                       listen_port=12346,
                                       transport="zmq")
    src_fa = bytearray(b"fa0123456789")
    src_gdn = bytearray(b"gdABCDEFGHIJ")
    dst_fa = bytearray(b"." * 16)
    dst_gdn = bytearray(b"?" * 16)

    producer.register_local_region(
        buffer=src_fa,
        region=RegisteredMemoryRegion(region_id="fa",
                                      nbytes=len(src_fa),
                                      page_bytes=12),
    )
    producer.register_local_region(
        buffer=src_gdn,
        region=RegisteredMemoryRegion(region_id="gdn",
                                      nbytes=len(src_gdn),
                                      page_bytes=12),
    )
    consumer.register_local_region(
        buffer=dst_fa,
        region=RegisteredMemoryRegion(region_id="fa",
                                      nbytes=len(dst_fa),
                                      page_bytes=16),
    )
    consumer.register_local_region(
        buffer=dst_gdn,
        region=RegisteredMemoryRegion(region_id="gdn",
                                      nbytes=len(dst_gdn),
                                      page_bytes=16),
    )

    metadata_dict = producer.local_metadata().to_dict()
    assert metadata_dict["regions"] == [
        {
            "region_id": "fa",
            "nbytes": len(src_fa),
            "page_bytes": 12,
        },
        {
            "region_id": "gdn",
            "nbytes": len(src_gdn),
            "page_bytes": 12,
        },
    ]

    compact = producer.gather_for_request([
        StridedTransferOp(src_region_id="gdn",
                          dst_region_id="gdn",
                          src_offset_bytes=2,
                          dst_offset_bytes=4,
                          segment_bytes=3,
                          src_stride_bytes=5,
                          dst_stride_bytes=4,
                          num_segments=2)
    ])
    assert compact == b"ABC" + b"FGH"
    session = consumer.new_destination_write_session()
    assert session.scatter(compact, [
        StridedTransferOp(src_region_id="gdn",
                          dst_region_id="gdn",
                          src_offset_bytes=2,
                          dst_offset_bytes=4,
                          segment_bytes=3,
                          src_stride_bytes=5,
                          dst_stride_bytes=4,
                          num_segments=2)
    ]) == 6
    session.flush_all()
    assert dst_fa == bytearray(b"." * 16)
    assert dst_gdn == bytearray(b"????ABC?FGH?????")


def test_wire_transfer_op_requires_region_ids() -> None:
    _require_api()
    with pytest.raises(KeyError):
        StridedTransferOp.from_mapping({
            "src_offset_bytes": 0,
            "dst_offset_bytes": 0,
            "segment_bytes": 1,
            "src_stride_bytes": 1,
            "dst_stride_bytes": 1,
            "num_segments": 1,
        })


def test_raw_region_registration_requires_explicit_descriptor() -> None:
    _require_api()
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=12345,
                                     transport="zmq")

    with pytest.raises(TypeError):
        RegisteredMemoryRegion(region_id="raw.layer", nbytes=8)
    with pytest.raises(TypeError):
        engine.register_local_region(bytearray(8),
                                     region_id="raw.layer",
                                     page_bytes=4)


@pytest.mark.parametrize("dtype_name", ["float8_e4m3fn", "bfloat16"])
def test_register_local_region_accepts_supported_typed_tensors(
        dtype_name) -> None:
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    tensor = torch.empty(32, dtype=dtype)
    region = _register_region(_engine(),
                              tensor,
                              region_id="typed",
                              page_bytes=tensor.nbytes)

    assert region.nbytes == tensor.nbytes


@pytest.mark.parametrize("dtype_name", ["int8", "uint8", "float32"])
def test_register_local_region_rejects_unsupported_tensor_dtype(
        dtype_name) -> None:
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    tensor = torch.zeros(32, dtype=dtype)

    with pytest.raises(TypeError, match="FP8/BF16"):
        _register_region(_engine(),
                         tensor,
                         region_id="typed",
                         page_bytes=tensor.nbytes)


def test_register_local_region_rejects_noncontiguous_typed_tensor() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.empty((4, 4), dtype=torch.bfloat16).transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous"):
        _register_region(_engine(),
                         tensor,
                         region_id="typed",
                         page_bytes=tensor.nbytes)


def test_destination_session_accepts_torch_fp8_tensor_buffers() -> None:
    _require_api()
    torch = pytest.importorskip("torch")
    src_bytes = _buffer(rank=9, size=96)
    dst = torch.zeros(128, dtype=torch.float8_e4m3fn)
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    _register_region(engine, dst, region_id="destination", page_bytes=64)
    ops = [
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination",
                          src_offset_bytes=3,
                          dst_offset_bytes=8,
                          segment_bytes=5,
                          src_stride_bytes=11,
                          dst_stride_bytes=13,
                          num_segments=4),
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination",
                          src_offset_bytes=70,
                          dst_offset_bytes=90,
                          segment_bytes=6,
                          src_stride_bytes=6,
                          dst_stride_bytes=8,
                          num_segments=2),
    ]

    compact = bytes(src_bytes[3:8] + src_bytes[14:19] + src_bytes[25:30] +
                    src_bytes[36:41] + src_bytes[70:76] + src_bytes[76:82])
    session = engine.new_destination_write_session()
    copied = session.scatter(compact, ops)
    session.flush_all()

    expected = bytearray(128)
    for op in ops:
        for i in range(op.num_segments):
            src_offset = op.src_offset_bytes + i * op.src_stride_bytes
            dst_offset = op.dst_offset_bytes + i * op.dst_stride_bytes
            expected[dst_offset:dst_offset +
                     op.segment_bytes] = src_bytes[src_offset:src_offset +
                                                   op.segment_bytes]
    assert copied == sum(op.segment_bytes * op.num_segments for op in ops)
    assert dst.view(torch.uint8).numpy().tobytes() == bytes(expected)


@pytest.mark.parametrize("dtype_name", ["float8_e4m3fn", "bfloat16"])
def test_destination_session_uses_one_whole_typed_tensor_host_cache(
        monkeypatch, dtype_name) -> None:
    _require_api()
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    initial = bytes(range(64))
    dst = torch.frombuffer(bytearray(initial), dtype=dtype).clone()
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    _register_region(engine, dst, region_id="destination", page_bytes=16)

    original_tensor_to_bytes = strided_transfer._tensor_to_bytes
    tensor_reads = []
    tensor_writes = []

    def tensor_to_bytes(tensor):
        payload = original_tensor_to_bytes(tensor)
        tensor_reads.append(payload)
        return payload

    def copy_bytes_to_tensor(payload, tensor):
        tensor_writes.append(bytes(payload))
        cpu_bytes = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        tensor.copy_(cpu_bytes.view(tensor.dtype).reshape(tensor.shape))

    monkeypatch.setattr(strided_transfer, "_tensor_to_bytes", tensor_to_bytes)
    monkeypatch.setattr(strided_transfer,
                        "_copy_bytes_to_tensor",
                        copy_bytes_to_tensor,
                        raising=False)

    session = engine.new_destination_write_session()
    first = [
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination",
                          src_offset_bytes=0,
                          dst_offset_bytes=18,
                          segment_bytes=4,
                          src_stride_bytes=4,
                          dst_stride_bytes=4,
                          num_segments=1)
    ]
    second = [
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination",
                          src_offset_bytes=0,
                          dst_offset_bytes=26,
                          segment_bytes=2,
                          src_stride_bytes=2,
                          dst_stride_bytes=2,
                          num_segments=1)
    ]

    assert session.scatter(b"abcd", first) == 4
    assert session.scatter(b"de", second) == 2
    session.flush_all()

    expected = bytearray(initial)
    expected[18:22] = b"abcd"
    expected[26:28] = b"de"
    assert dst.view(torch.uint8).numpy().tobytes() == bytes(expected)
    assert tensor_reads == [initial]
    assert tensor_writes == [bytes(expected)]


def test_overlapping_destination_sessions_share_lock_per_raw_tensor(
        monkeypatch) -> None:
    _require_api()

    class FakeTensor:

        def __init__(self, nbytes: int) -> None:
            self.data = bytearray(nbytes)
            self.nbytes = nbytes

    dst = FakeTensor(16)
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    _register_region(engine, dst, region_id="destination.first", page_bytes=4)
    _register_region(engine, dst, region_id="destination.second", page_bytes=4)
    monkeypatch.setattr(strided_transfer, "_is_torch_tensor",
                        lambda buffer: isinstance(buffer, FakeTensor))

    tensor_reads = []

    def tensor_to_bytes(tensor):
        payload = bytes(tensor.data)
        tensor_reads.append(payload)
        return payload

    def copy_bytes_to_fake_tensor(payload, tensor):
        tensor.data[:] = payload

    monkeypatch.setattr(strided_transfer, "_tensor_to_bytes", tensor_to_bytes)
    monkeypatch.setattr(strided_transfer,
                        "_copy_bytes_to_tensor",
                        copy_bytes_to_fake_tensor,
                        raising=False)

    first = [
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination.first",
                          src_offset_bytes=0,
                          dst_offset_bytes=0,
                          segment_bytes=4,
                          src_stride_bytes=4,
                          dst_stride_bytes=4,
                          num_segments=1)
    ]
    second = [
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination.second",
                          src_offset_bytes=0,
                          dst_offset_bytes=8,
                          segment_bytes=4,
                          src_stride_bytes=4,
                          dst_stride_bytes=4,
                          num_segments=1)
    ]

    session1 = engine.new_destination_write_session()
    session2 = engine.new_destination_write_session()
    assert session1.scatter(b"1111", first) == 4

    second_started = threading.Event()
    second_done = threading.Event()
    second_error = []

    def run_second_scatter():
        second_started.set()
        try:
            assert session2.scatter(b"2222", second) == 4
        except Exception as exc:
            second_error.append(exc)
        finally:
            second_done.set()

    thread = threading.Thread(target=run_second_scatter)
    thread.start()
    assert second_started.wait(timeout=1.0)
    assert not second_done.wait(timeout=0.1)

    session1.flush_all()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert not second_error
    session2.flush_all()

    expected = bytearray(16)
    expected[0:4] = b"1111"
    expected[8:12] = b"2222"
    assert dst.data == expected
    assert tensor_reads == [
        bytes(bytearray(16)),
        bytes(expected[:4] + bytearray(12)),
    ]


def test_tpu_tensor_read_keeps_existing_torch_slice_path() -> None:
    _require_api()
    torch = pytest.importorskip("torch")
    payload = bytes(range(64))
    tensor = torch.frombuffer(bytearray(payload),
                              dtype=torch.float8_e4m3fn).clone()

    assert strided_transfer._read_bytes(tensor, 32, 16) == bytes(range(32, 48))


def test_torch_tensor_source_read_preserves_typed_bytes() -> None:
    _require_api()
    torch = pytest.importorskip("torch")
    dtypes = [
        getattr(torch, "float8_e4m3fn", None),
        torch.bfloat16,
    ]
    payload = bytes(range(16))
    for dtype in dtypes:
        if dtype is None:
            continue
        tensor = torch.frombuffer(bytearray(payload), dtype=dtype).clone()
        assert strided_transfer._tensor_slice_to_bytes(tensor, 0,
                                                       len(payload)) == payload


def test_torch_tensor_source_read_rejects_unaligned_typed_range() -> None:
    _require_api()
    torch = pytest.importorskip("torch")
    tensor = torch.zeros(8, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="aligned to tensor element_size"):
        strided_transfer._tensor_slice_to_bytes(tensor, 1, 4)


def test_remote_metadata_wraps_worker_identity_and_zmq_endpoint() -> None:
    _require_api()

    metadata = RemoteWorkerMetadata(dp_rank=2,
                                    tp_rank=5,
                                    worker_id="prefill-dp2-tp5",
                                    tcp_host="10.0.0.8",
                                    tcp_port=32005,
                                    transport="zmq",
                                    regions=())

    assert metadata.to_dict() == {
        "dp_rank": 2,
        "tp_rank": 5,
        "worker_id": "prefill-dp2-tp5",
        "tcp_host": "10.0.0.8",
        "tcp_port": 32005,
        "transport": "zmq",
        "regions": [],
    }
    assert "tpu_worker_host" not in metadata.to_dict()
    assert "tpu_worker_port" not in metadata.to_dict()


def test_local_metadata_carries_registered_raw_regions() -> None:
    _require_api()
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=3,
                                     local_worker_id="prefill-rank3",
                                     listen_host="127.0.0.1",
                                     listen_port=32003,
                                     transport="zmq")
    _register_region(engine,
                     bytearray(16),
                     region_id="model.layers.0.self_attn",
                     page_bytes=8)
    _register_region(engine,
                     bytearray(32),
                     region_id="model.layers.1.linear_attn.state0",
                     page_bytes=16)

    assert engine.local_metadata().to_dict()["regions"] == [
        {
            "region_id": "model.layers.0.self_attn",
            "nbytes": 16,
            "page_bytes": 8,
        },
        {
            "region_id": "model.layers.1.linear_attn.state0",
            "nbytes": 32,
            "page_bytes": 16,
        },
    ]


def test_raw_regions_use_region_relative_offsets_over_zmq() -> None:
    _require_api()
    src_fa = bytearray(b"abcdefghijklmnop")
    src_gdn = bytearray(b"ABCDEFGHIJKLMNOP")
    dst_fa = bytearray(b"." * 20)
    dst_gdn = bytearray(b"?" * 20)
    producer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=2,
                                       local_worker_id="producer",
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    consumer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=0,
                                       local_worker_id="consumer",
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    _register_region(producer, src_fa, region_id="fa", page_bytes=8)
    _register_region(producer, src_gdn, region_id="gdn", page_bytes=8)
    _register_region(consumer, dst_fa, region_id="fa", page_bytes=10)
    _register_region(consumer, dst_gdn, region_id="gdn", page_bytes=10)
    producer.start()
    try:
        metadata = producer.local_metadata()
        metadata_dict = metadata.to_dict()
        assert metadata_dict["regions"] == [
            {
                "region_id": "fa",
                "nbytes": 16,
                "page_bytes": 8,
            },
            {
                "region_id": "gdn",
                "nbytes": 16,
                "page_bytes": 8,
            },
        ]
        consumer.register_other_remote_metadata([metadata])

        ops = [
            StridedTransferOp(src_region_id="fa",
                              dst_region_id="fa",
                              src_offset_bytes=1,
                              dst_offset_bytes=2,
                              segment_bytes=3,
                              src_stride_bytes=7,
                              dst_stride_bytes=4,
                              num_segments=2),
            StridedTransferOp(src_region_id="gdn",
                              dst_region_id="gdn",
                              src_offset_bytes=4,
                              dst_offset_bytes=1,
                              segment_bytes=4,
                              src_stride_bytes=6,
                              dst_stride_bytes=7,
                              num_segments=2),
        ]

        session = consumer.new_destination_write_session()
        copied = consumer.pull_from_registered_into_session(
            remote_tp_rank=2, dp_rank=0, ops=ops, write_session=session)
        session.flush_all()

        assert copied == 14
        assert dst_fa == bytearray(b"..bcd.ijk...........")
        assert dst_gdn == bytearray(b"?EFGH???KLMN????????")
    finally:
        producer.stop()
        consumer.stop()


def test_region_gather_caches_reused_source_pages() -> None:
    _require_api()

    class CountingBuffer:

        def __init__(self, payload: bytes):
            self.payload = payload
            self.nbytes = len(payload)
            self.reads = []

        def read_bytes(self, offset: int, length: int) -> bytes:
            self.reads.append((offset, length))
            return self.payload[offset:offset + length]

    source = CountingBuffer(bytes(range(64)))
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    _register_region(engine, source, region_id="page.cached", page_bytes=16)
    ops = [
        StridedTransferOp(src_region_id="page.cached",
                          dst_region_id="destination",
                          src_offset_bytes=1,
                          dst_offset_bytes=0,
                          segment_bytes=3,
                          src_stride_bytes=4,
                          dst_stride_bytes=3,
                          num_segments=2),
        StridedTransferOp(src_region_id="page.cached",
                          dst_region_id="destination",
                          src_offset_bytes=7,
                          dst_offset_bytes=6,
                          segment_bytes=4,
                          src_stride_bytes=2,
                          dst_stride_bytes=4,
                          num_segments=1),
    ]

    compact = engine.gather_for_request(ops)

    assert compact == source.payload[1:4] + source.payload[
        5:8] + source.payload[7:11]
    assert source.reads == [(0, 16)]


def test_region_gather_rejects_segment_that_crosses_page_boundary() -> None:
    _require_api()
    source = bytearray(b"abcdefghijklmnop")
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    _register_region(engine, source, region_id="page.strict", page_bytes=8)
    op = StridedTransferOp(src_region_id="page.strict",
                           dst_region_id="destination",
                           src_offset_bytes=6,
                           dst_offset_bytes=0,
                           segment_bytes=4,
                           src_stride_bytes=4,
                           dst_stride_bytes=4,
                           num_segments=1)

    with pytest.raises(ValueError, match="crosses source page boundary"):
        engine.gather_for_request([op])


def test_region_scatter_caches_only_touched_destination_pages() -> None:
    _require_api()

    class CountingWritableBuffer:

        def __init__(self, payload: bytes):
            self.payload = bytearray(payload)
            self.nbytes = len(payload)
            self.reads = []
            self.writes = []

        def read_bytes(self, offset: int, length: int) -> bytes:
            self.reads.append((offset, length))
            return bytes(self.payload[offset:offset + length])

        def write_bytes(self, offset: int, data) -> None:
            data = bytes(memoryview(data).cast("B"))
            self.writes.append((offset, len(data)))
            self.payload[offset:offset + len(data)] = data

    destination = CountingWritableBuffer(bytes(range(64)))
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    _register_region(engine,
                     destination,
                     region_id="destination",
                     page_bytes=16)
    compact = b"abcde"
    ops = [
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination",
                          src_offset_bytes=0,
                          dst_offset_bytes=18,
                          segment_bytes=3,
                          src_stride_bytes=3,
                          dst_stride_bytes=5,
                          num_segments=1),
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination",
                          src_offset_bytes=3,
                          dst_offset_bytes=22,
                          segment_bytes=2,
                          src_stride_bytes=2,
                          dst_stride_bytes=2,
                          num_segments=1),
    ]

    session = engine.new_destination_write_session()
    copied = session.scatter(compact, ops)
    session.flush_all()

    expected = bytearray(range(64))
    expected[18:21] = b"abc"
    expected[22:24] = b"de"
    assert copied == 5
    assert destination.payload == expected
    assert destination.reads == [(16, 16)]
    assert destination.writes == [(16, 16)]


def test_destination_page_write_session_reuses_pages_across_scatters() -> None:
    _require_api()

    class CountingWritableBuffer:

        def __init__(self, payload: bytes):
            self.payload = bytearray(payload)
            self.nbytes = len(payload)
            self.reads = []
            self.writes = []

        def read_bytes(self, offset: int, length: int) -> bytes:
            self.reads.append((offset, length))
            return bytes(self.payload[offset:offset + length])

        def write_bytes(self, offset: int, data) -> None:
            data = bytes(memoryview(data).cast("B"))
            self.writes.append((offset, len(data)))
            self.payload[offset:offset + len(data)] = data

    destination = CountingWritableBuffer(bytes(range(64)))
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    _register_region(engine,
                     destination,
                     region_id="destination",
                     page_bytes=16)
    session = engine.new_destination_write_session()

    first = [
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination",
                          src_offset_bytes=0,
                          dst_offset_bytes=18,
                          segment_bytes=3,
                          src_stride_bytes=3,
                          dst_stride_bytes=3,
                          num_segments=1)
    ]
    second = [
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination",
                          src_offset_bytes=0,
                          dst_offset_bytes=22,
                          segment_bytes=2,
                          src_stride_bytes=2,
                          dst_stride_bytes=2,
                          num_segments=1)
    ]

    assert session.scatter(b"abc", first) == 3
    assert session.scatter(b"de", second) == 2
    session.flush_all()

    expected = bytearray(range(64))
    expected[18:21] = b"abc"
    expected[22:24] = b"de"
    assert destination.payload == expected
    assert destination.reads == [(16, 16)]
    assert destination.writes == [(16, 16)]


def test_register_other_remote_metadata_requires_explicit_dp_rank() -> None:
    _require_api()
    engine = StridedKVTransferEngine(local_dp_rank=1,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    engine.register_other_remote_metadata([
        RemoteWorkerMetadata(dp_rank=0,
                             tp_rank=3,
                             worker_id="dp0-tp3",
                             tcp_host="127.0.0.1",
                             tcp_port=11000,
                             transport="zmq",
                             regions=()),
        RemoteWorkerMetadata(dp_rank=1,
                             tp_rank=3,
                             worker_id="dp1-tp3",
                             tcp_host="127.0.0.2",
                             tcp_port=11001,
                             transport="zmq",
                             regions=()),
    ])

    with pytest.raises(TypeError):
        engine.resolve_remote(tp_rank=3)
    assert engine.resolve_remote(tp_rank=3, dp_rank=0).worker_id == "dp0-tp3"
    assert engine.resolve_remote(tp_rank=3, dp_rank=0).tcp_host == "127.0.0.1"
    assert engine.resolve_remote(tp_rank=3, dp_rank=1).worker_id == "dp1-tp3"
    assert engine.resolve_remote(tp_rank=3, dp_rank=1).tcp_host == "127.0.0.2"

    with pytest.raises(KeyError, match="dp_rank=1 tp_rank=9"):
        engine.resolve_remote(tp_rank=9, dp_rank=1)


def test_remote_metadata_rejects_non_zmq_transport() -> None:
    _require_api()

    with pytest.raises(ValueError, match="transport must be 'zmq'"):
        RemoteWorkerMetadata(dp_rank=0,
                             tp_rank=0,
                             worker_id="p0",
                             tcp_host="127.0.0.1",
                             tcp_port=12345,
                             transport="tcp",
                             regions=())


def test_runtime_api_rejects_mapping_shims() -> None:
    _require_api()
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    region = RegisteredMemoryRegion(region_id="source",
                                    nbytes=16,
                                    page_bytes=16)
    op = StridedTransferOp(src_region_id="source",
                           dst_region_id="destination",
                           src_offset_bytes=0,
                           dst_offset_bytes=0,
                           segment_bytes=1,
                           src_stride_bytes=1,
                           dst_stride_bytes=1,
                           num_segments=1)
    remote = RemoteWorkerMetadata(dp_rank=0,
                                  tp_rank=1,
                                  worker_id="p0-tp1",
                                  tcp_host="127.0.0.1",
                                  tcp_port=_unused_tcp_port(),
                                  transport="zmq",
                                  regions=())

    with pytest.raises(TypeError, match="RegisteredMemoryRegion"):
        engine.register_local_region(buffer=bytearray(16),
                                     region=region.to_dict())
    engine.register_local_region(buffer=bytearray(16), region=region)

    with pytest.raises(TypeError, match="RegisteredMemoryRegion"):
        RemoteWorkerMetadata(dp_rank=0,
                             tp_rank=2,
                             worker_id="p0-tp2",
                             tcp_host="127.0.0.1",
                             tcp_port=12345,
                             transport="zmq",
                             regions=(region.to_dict(), ))
    with pytest.raises(TypeError, match="RemoteWorkerMetadata"):
        engine.register_other_remote_metadata(remote)
    with pytest.raises(TypeError, match="RemoteWorkerMetadata"):
        engine.register_other_remote_metadata([remote.to_dict()])
    engine.register_other_remote_metadata([remote])

    with pytest.raises(TypeError, match="StridedTransferOp"):
        engine.gather_for_request([op.to_dict()])
    with pytest.raises(TypeError, match="RemoteWorkerMetadata"):
        engine.pull_into_session(remote.to_dict(), [op], object())
    with pytest.raises(TypeError, match="StridedTransferOp"):
        engine.pull_from_registered_into_session(remote_tp_rank=1,
                                                 dp_rank=0,
                                                 ops=[op.to_dict()],
                                                 write_session=object())


def test_registration_and_resolve_do_not_connect_until_read() -> None:
    _require_api()
    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    metadata = RemoteWorkerMetadata(dp_rank=0,
                                    tp_rank=2,
                                    worker_id="unstarted",
                                    tcp_host="127.0.0.1",
                                    tcp_port=_unused_tcp_port(),
                                    transport="zmq",
                                    regions=())

    engine.register_other_remote_metadata([metadata])
    assert engine.resolve_remote(tp_rank=2, dp_rank=0) == metadata
    _register_region(engine,
                     bytearray(8),
                     region_id="destination",
                     page_bytes=8)
    session = engine.new_destination_write_session()

    with pytest.raises(TimeoutError):
        engine.pull_from_registered_into_session(
            remote_tp_rank=2,
            dp_rank=0,
            ops=[
                StridedTransferOp(src_region_id="source",
                                  dst_region_id="destination",
                                  src_offset_bytes=0,
                                  dst_offset_bytes=0,
                                  segment_bytes=1,
                                  src_stride_bytes=1,
                                  dst_stride_bytes=1,
                                  num_segments=1)
            ],
            write_session=session,
        )


def test_engine_pulls_from_registered_remote_worker_over_zmq() -> None:
    _require_api()
    src = _buffer(rank=4, size=256)
    dst = bytearray(300)
    producer = _make_zmq_producer(dp_rank=0,
                                  tp_rank=2,
                                  worker_id="p0-tp2",
                                  source=src)
    consumer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=0,
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    _register_region(consumer, dst, region_id="destination", page_bytes=100)
    try:
        metadata = producer.local_metadata()
        assert metadata.worker_id == "p0-tp2"
        consumer.register_other_remote_metadata([metadata])
        ops = [
            StridedTransferOp(src_region_id="source",
                              dst_region_id="destination",
                              src_offset_bytes=3,
                              dst_offset_bytes=10,
                              segment_bytes=5,
                              src_stride_bytes=13,
                              dst_stride_bytes=17,
                              num_segments=4),
            StridedTransferOp(src_region_id="source",
                              dst_region_id="destination",
                              src_offset_bytes=120,
                              dst_offset_bytes=150,
                              segment_bytes=8,
                              src_stride_bytes=8,
                              dst_stride_bytes=9,
                              num_segments=2),
        ]

        session = consumer.new_destination_write_session()
        bytes_copied = consumer.pull_from_registered_into_session(
            remote_tp_rank=2, dp_rank=0, ops=ops, write_session=session)
        session.flush_all()

        assert bytes_copied == 36
        _assert_segments_match(src, dst, ops)
    finally:
        producer.stop()


def test_zmq_pull_requires_registered_source_regions() -> None:
    _require_api()
    producer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=2,
                                       local_worker_id="p0-tp2",
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    consumer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=0,
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    _register_region(consumer,
                     bytearray(8),
                     region_id="destination",
                     page_bytes=8)
    producer.start()
    try:
        consumer.register_other_remote_metadata([producer.local_metadata()])
        session = consumer.new_destination_write_session()
        with pytest.raises(RuntimeError,
                           match="source regions are not registered"):
            consumer.pull_from_registered_into_session(
                remote_tp_rank=2,
                dp_rank=0,
                ops=[
                    StridedTransferOp(src_region_id="source",
                                      dst_region_id="destination",
                                      src_offset_bytes=0,
                                      dst_offset_bytes=0,
                                      segment_bytes=1,
                                      src_stride_bytes=1,
                                      dst_stride_bytes=1,
                                      num_segments=1)
                ],
                write_session=session,
            )
    finally:
        producer.stop()


def test_multi_worker_pcp_regroup_4pcp_to_1tp_over_zmq() -> None:
    _require_api()
    remote_pcp = 4
    remote_block_size = 2
    local_block_size = 8
    token_bytes = 6
    remote_page_bytes = remote_block_size * token_bytes
    local_page_bytes = local_block_size * token_bytes
    local_block_id = 1
    remote_block_ids = [1, 3, 0, 2]

    consumer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=0,
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    producers = []
    dst = bytearray(3 * local_page_bytes)
    expected = bytearray(len(dst))
    _register_region(consumer,
                     dst,
                     region_id="destination",
                     page_bytes=local_page_bytes)
    try:
        session = consumer.new_destination_write_session()
        for pcp_rank in range(remote_pcp):
            source = _buffer(rank=10 + pcp_rank, size=5 * remote_page_bytes)
            producer = _make_zmq_producer(dp_rank=0,
                                          tp_rank=pcp_rank,
                                          source=source)
            producers.append(producer)
            consumer.register_other_remote_metadata(
                [producer.local_metadata()])

            op = StridedTransferOp(
                src_region_id="source",
                dst_region_id="destination",
                src_offset_bytes=remote_block_ids[pcp_rank] *
                remote_page_bytes,
                dst_offset_bytes=(local_block_id * local_page_bytes +
                                  pcp_rank * remote_block_size * token_bytes),
                segment_bytes=token_bytes,
                src_stride_bytes=token_bytes,
                dst_stride_bytes=token_bytes,
                num_segments=remote_block_size)
            assert _apply_expected_strided_ops(
                source, expected, [op]) == (remote_block_size * token_bytes)

            copied = consumer.pull_from_registered_into_session(
                remote_tp_rank=pcp_rank,
                dp_rank=0,
                ops=[op],
                write_session=session)
            assert copied == remote_block_size * token_bytes

        session.flush_all()
        assert dst == expected
    finally:
        for producer in producers:
            producer.stop()


def test_multi_worker_tp_merge_4tp_to_1tp_over_zmq() -> None:
    _require_api()
    remote_tp = 4
    tokens_per_page = 3
    head_bytes = 5
    remote_page_bytes = tokens_per_page * head_bytes
    local_token_stride = remote_tp * head_bytes
    local_page_bytes = tokens_per_page * local_token_stride
    local_block_id = 2

    consumer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=0,
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    producers = []
    dst = bytearray(4 * local_page_bytes)
    expected = bytearray(len(dst))
    _register_region(consumer,
                     dst,
                     region_id="destination",
                     page_bytes=local_page_bytes)
    try:
        session = consumer.new_destination_write_session()
        for remote_rank in range(remote_tp):
            source = _buffer(rank=30 + remote_rank, size=5 * remote_page_bytes)
            producer = _make_zmq_producer(dp_rank=0,
                                          tp_rank=remote_rank,
                                          source=source)
            producers.append(producer)
            consumer.register_other_remote_metadata(
                [producer.local_metadata()])

            op = StridedTransferOp(
                src_region_id="source",
                dst_region_id="destination",
                src_offset_bytes=remote_rank * remote_page_bytes,
                dst_offset_bytes=(local_block_id * local_page_bytes +
                                  remote_rank * head_bytes),
                segment_bytes=head_bytes,
                src_stride_bytes=head_bytes,
                dst_stride_bytes=local_token_stride,
                num_segments=tokens_per_page)
            assert _apply_expected_strided_ops(
                source, expected, [op]) == tokens_per_page * head_bytes

            copied = consumer.pull_from_registered_into_session(
                remote_tp_rank=remote_rank,
                dp_rank=0,
                ops=[op],
                write_session=session)
            assert copied == tokens_per_page * head_bytes

        session.flush_all()
        assert dst == expected
    finally:
        for producer in producers:
            producer.stop()


@pytest.mark.parametrize(("local_rank", "remote_rank", "src_head_index"), [
    (0, 0, 0),
    (1, 0, 1),
    (2, 1, 0),
    (3, 1, 1),
])
def test_tp_split_2tp_to_4tp_reads_only_local_head_slice_over_zmq(
        local_rank: int, remote_rank: int, src_head_index: int) -> None:
    _require_api()
    tokens_per_page = 3
    head_bytes = 7
    remote_heads_per_rank = 2
    local_heads_per_rank = 1
    remote_page_bytes = tokens_per_page * remote_heads_per_rank * head_bytes
    local_page_bytes = tokens_per_page * local_heads_per_rank * head_bytes
    remote_block_id = 2
    local_block_id = 1 + local_rank

    source = _buffer(rank=50 + remote_rank, size=4 * remote_page_bytes)
    producer = _make_zmq_producer(dp_rank=0,
                                  tp_rank=remote_rank,
                                  source=source)
    consumer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=local_rank,
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    dst = bytearray(6 * local_page_bytes)
    expected = bytearray(len(dst))
    _register_region(consumer,
                     dst,
                     region_id="destination",
                     page_bytes=local_page_bytes)
    try:
        consumer.register_other_remote_metadata([producer.local_metadata()])
        session = consumer.new_destination_write_session()
        op = StridedTransferOp(
            src_region_id="source",
            dst_region_id="destination",
            src_offset_bytes=(remote_block_id * remote_page_bytes +
                              src_head_index * head_bytes),
            dst_offset_bytes=local_block_id * local_page_bytes,
            segment_bytes=head_bytes,
            src_stride_bytes=remote_heads_per_rank * head_bytes,
            dst_stride_bytes=local_heads_per_rank * head_bytes,
            num_segments=tokens_per_page)
        assert _apply_expected_strided_ops(
            source, expected, [op]) == tokens_per_page * head_bytes

        copied = consumer.pull_from_registered_into_session(
            remote_tp_rank=remote_rank,
            dp_rank=0,
            ops=[op],
            write_session=session)
        session.flush_all()

        assert copied == tokens_per_page * head_bytes
        assert dst == expected
    finally:
        producer.stop()


def test_pull_from_registered_session_uses_dp_ego_for_actual_zmq_transfer(
) -> None:
    _require_api()
    op_default = StridedTransferOp(src_region_id="source",
                                   dst_region_id="default",
                                   src_offset_bytes=4,
                                   dst_offset_bytes=9,
                                   segment_bytes=5,
                                   src_stride_bytes=12,
                                   dst_stride_bytes=7,
                                   num_segments=4)
    op_explicit = StridedTransferOp(src_region_id="source",
                                    dst_region_id="explicit",
                                    src_offset_bytes=4,
                                    dst_offset_bytes=9,
                                    segment_bytes=5,
                                    src_stride_bytes=12,
                                    dst_stride_bytes=7,
                                    num_segments=4)
    producer_dp0_src = _buffer(rank=70, size=96)
    producer_dp1_src = _buffer(rank=71, size=96)
    producer_dp0 = _make_zmq_producer(dp_rank=0,
                                      tp_rank=0,
                                      worker_id="dp0-tp0",
                                      source=producer_dp0_src)
    producer_dp1 = _make_zmq_producer(dp_rank=1,
                                      tp_rank=0,
                                      worker_id="dp1-tp0",
                                      source=producer_dp1_src)
    consumer = StridedKVTransferEngine(local_dp_rank=1,
                                       local_tp_rank=0,
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    try:
        consumer.register_other_remote_metadata(
            [producer_dp0.local_metadata(),
             producer_dp1.local_metadata()])

        default_dst = bytearray(64)
        explicit_dst = bytearray(64)
        _register_region(consumer,
                         default_dst,
                         region_id="default",
                         page_bytes=64)
        _register_region(consumer,
                         explicit_dst,
                         region_id="explicit",
                         page_bytes=64)
        default_expected = bytearray(64)
        _apply_expected_strided_ops(producer_dp1_src, default_expected,
                                    [op_default])
        session = consumer.new_destination_write_session()
        with pytest.raises(TypeError):
            consumer.pull_from_registered_into_session(remote_tp_rank=0,
                                                       ops=[op_default],
                                                       write_session=session)
        assert consumer.pull_from_registered_into_session(
            remote_tp_rank=0,
            dp_rank=1,
            ops=[op_default],
            write_session=session) == 20
        session.flush_all()
        assert default_dst == default_expected

        explicit_expected = bytearray(64)
        _apply_expected_strided_ops(producer_dp0_src, explicit_expected,
                                    [op_explicit])
        session = consumer.new_destination_write_session()
        assert consumer.pull_from_registered_into_session(
            remote_tp_rank=0,
            dp_rank=0,
            ops=[op_explicit],
            write_session=session) == 20
        session.flush_all()
        assert explicit_dst == explicit_expected
        assert explicit_dst != default_dst
    finally:
        producer_dp0.stop()
        producer_dp1.stop()


def test_pull_uses_the_runtime_remote_worker_descriptor() -> None:
    _require_api()
    producer0_src = _buffer(rank=0, size=128)
    producer1_src = _buffer(rank=1, size=128)
    producer0 = _make_zmq_producer(dp_rank=0,
                                   tp_rank=0,
                                   worker_id="rank0",
                                   source=producer0_src)
    producer1 = _make_zmq_producer(dp_rank=0,
                                   tp_rank=1,
                                   worker_id="rank1",
                                   source=producer1_src)
    consumer = StridedKVTransferEngine(local_dp_rank=0,
                                       local_tp_rank=0,
                                       listen_host="127.0.0.1",
                                       listen_port=0,
                                       transport="zmq")
    dst = bytearray(64)
    _register_region(consumer, dst, region_id="destination", page_bytes=64)
    try:
        consumer.register_other_remote_metadata(
            [producer0.local_metadata(),
             producer1.local_metadata()])
        ops = [
            StridedTransferOp(src_region_id="source",
                              dst_region_id="destination",
                              src_offset_bytes=7,
                              dst_offset_bytes=5,
                              segment_bytes=3,
                              src_stride_bytes=11,
                              dst_stride_bytes=4,
                              num_segments=5)
        ]
        session = consumer.new_destination_write_session()

        bytes_copied = consumer.pull_into_session(producer1.local_metadata(),
                                                  ops, session)
        session.flush_all()

        assert bytes_copied == 15
        _assert_segments_match(producer1_src, dst, ops)
        assert dst[5:8] != producer0_src[7:10]
    finally:
        producer0.stop()
        producer1.stop()


def test_invalid_op_and_out_of_bounds_ranges_raise_value_error() -> None:
    _require_api()
    with pytest.raises(ValueError, match="segment_bytes must be positive"):
        StridedTransferOp(src_region_id="source",
                          dst_region_id="destination",
                          src_offset_bytes=0,
                          dst_offset_bytes=0,
                          segment_bytes=0,
                          src_stride_bytes=1,
                          dst_stride_bytes=1,
                          num_segments=1)

    engine = StridedKVTransferEngine(local_dp_rank=0,
                                     local_tp_rank=0,
                                     listen_host="127.0.0.1",
                                     listen_port=0,
                                     transport="zmq")
    _register_region(engine,
                     bytearray(8),
                     region_id="destination",
                     page_bytes=8)
    session = engine.new_destination_write_session()
    with pytest.raises(ValueError, match="destination range out of bounds"):
        session.scatter(
            b"xx",
            [
                StridedTransferOp(src_region_id="source",
                                  dst_region_id="destination",
                                  src_offset_bytes=0,
                                  dst_offset_bytes=7,
                                  segment_bytes=2,
                                  src_stride_bytes=1,
                                  dst_stride_bytes=1,
                                  num_segments=1)
            ],
        )
