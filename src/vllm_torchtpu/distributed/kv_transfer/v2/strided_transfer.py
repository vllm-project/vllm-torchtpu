# SPDX-License-Identifier: Apache-2.0
"""Byte-op based strided KV/state transfer helpers.

This module deliberately does not build heterogeneous TP/PCP/GDN transfer
plans.  The caller owns layout-specific planning and passes a concrete op list
whose offsets are already relative to the source and destination buffers.

Execution is split the same way the real P/D transfer is split:

* P side gathers source strided ranges into a compact payload.
* D side receives that compact payload and scatters it into destination ranges.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import zmq

_PROTOCOL_VERSION = 1
_DEFAULT_MAX_FRAME_BYTES = 1 << 30
_MSG_PULL = b"STRIDED_PULL"
_MSG_OK = b"STRIDED_OK"
_MSG_ERROR = b"STRIDED_ERROR"


@dataclass(frozen=True)
class StridedTransferOp:
    src_region_id: str
    dst_region_id: str
    src_offset_bytes: int
    dst_offset_bytes: int
    segment_bytes: int
    src_stride_bytes: int
    dst_stride_bytes: int
    num_segments: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "src_region_id", str(self.src_region_id))
        object.__setattr__(self, "dst_region_id", str(self.dst_region_id))
        if not self.src_region_id:
            raise ValueError("src_region_id must be non-empty")
        if not self.dst_region_id:
            raise ValueError("dst_region_id must be non-empty")
        for field_name in (
                "src_offset_bytes",
                "dst_offset_bytes",
                "segment_bytes",
                "src_stride_bytes",
                "dst_stride_bytes",
                "num_segments",
        ):
            object.__setattr__(self, field_name, int(getattr(self,
                                                             field_name)))
        _require_non_negative("src_offset_bytes", self.src_offset_bytes)
        _require_non_negative("dst_offset_bytes", self.dst_offset_bytes)
        _require_positive("segment_bytes", self.segment_bytes)
        _require_non_negative("src_stride_bytes", self.src_stride_bytes)
        _require_non_negative("dst_stride_bytes", self.dst_stride_bytes)
        _require_positive("num_segments", self.num_segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src_region_id": self.src_region_id,
            "dst_region_id": self.dst_region_id,
            "src_offset_bytes": self.src_offset_bytes,
            "dst_offset_bytes": self.dst_offset_bytes,
            "segment_bytes": self.segment_bytes,
            "src_stride_bytes": self.src_stride_bytes,
            "dst_stride_bytes": self.dst_stride_bytes,
            "num_segments": self.num_segments,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StridedTransferOp":
        return cls(src_region_id=str(data["src_region_id"]),
                   dst_region_id=str(data["dst_region_id"]),
                   src_offset_bytes=int(data["src_offset_bytes"]),
                   dst_offset_bytes=int(data["dst_offset_bytes"]),
                   segment_bytes=int(data["segment_bytes"]),
                   src_stride_bytes=int(data["src_stride_bytes"]),
                   dst_stride_bytes=int(data["dst_stride_bytes"]),
                   num_segments=int(data["num_segments"]))


@dataclass(frozen=True)
class RegisteredMemoryRegion:
    region_id: str
    nbytes: int
    page_bytes: int

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must be non-empty")
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "nbytes", int(self.nbytes))
        object.__setattr__(self, "page_bytes", int(self.page_bytes))
        _require_positive("nbytes", self.nbytes)
        _require_positive("page_bytes", self.page_bytes)

    @property
    def effective_page_bytes(self) -> int:
        return self.page_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "nbytes": self.nbytes,
            "page_bytes": self.page_bytes,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RegisteredMemoryRegion":
        return cls(region_id=data["region_id"],
                   nbytes=int(data["nbytes"]),
                   page_bytes=int(data["page_bytes"]))


@dataclass(frozen=True)
class _LocalMemoryRegion:
    metadata: RegisteredMemoryRegion
    buffer: Any
    write_lock: threading.Lock = field(default_factory=threading.Lock,
                                       compare=False,
                                       repr=False)


@dataclass
class _CachedDestinationPage:
    region: _LocalMemoryRegion
    page_offset: int
    data: bytearray
    dirty: bool = False


@dataclass(frozen=True)
class RemoteWorkerMetadata:
    dp_rank: int
    tp_rank: int
    tcp_host: str
    tcp_port: int
    transport: str
    worker_id: str | None
    regions: tuple[RegisteredMemoryRegion, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dp_rank", int(self.dp_rank))
        object.__setattr__(self, "tp_rank", int(self.tp_rank))
        object.__setattr__(self, "tcp_port", int(self.tcp_port))
        object.__setattr__(self, "transport", self.transport.lower())
        object.__setattr__(self, "regions",
                           _require_registered_regions(self.regions))
        _require_non_negative("dp_rank", self.dp_rank)
        _require_non_negative("tp_rank", self.tp_rank)
        _require_positive("tcp_port", self.tcp_port)
        if not self.tcp_host:
            raise ValueError("tcp_host must be non-empty")
        if self.transport != "zmq":
            raise ValueError("transport must be 'zmq', got "
                             f"{self.transport!r}")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "worker_id": self.worker_id,
            "tcp_host": self.tcp_host,
            "tcp_port": self.tcp_port,
            "transport": self.transport,
            "regions": [region.to_dict() for region in self.regions],
        }
        return result

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RemoteWorkerMetadata":
        return cls(dp_rank=int(data["dp_rank"]),
                   tp_rank=int(data["tp_rank"]),
                   worker_id=data["worker_id"],
                   tcp_host=str(data["tcp_host"]),
                   tcp_port=int(data["tcp_port"]),
                   transport=str(data["transport"]),
                   regions=_registered_regions_from_wire(data["regions"]))


class StridedKVTransferEngine:
    """Per-worker Python reference engine for strided KV/state transfer.

    One instance represents one TPU worker.  It can serve P-side gathers from
    explicitly registered raw regions and can act as a D-side client that pulls
    compact payloads from explicitly selected remote workers.
    """

    def __init__(
        self,
        *,
        local_dp_rank: int,
        local_tp_rank: int,
        listen_host: str,
        listen_port: int,
        transport: str,
        local_worker_id: str | None = None,
        timeout_s: float = 30.0,
        max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        self.local_dp_rank = int(local_dp_rank)
        self.local_tp_rank = int(local_tp_rank)
        self.local_worker_id = local_worker_id
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        self.transport = transport.lower()
        self.timeout_s = float(timeout_s)
        self.max_frame_bytes = int(max_frame_bytes)
        self._local_regions: list[_LocalMemoryRegion] = []
        self._local_region_locks_by_buffer: dict[int, threading.Lock] = {}
        self._remote_metadata: dict[tuple[int, int], RemoteWorkerMetadata] = {}
        self._lock = threading.RLock()
        self._zmq_context = zmq.Context(io_threads=1)
        self._server_socket: zmq.Socket | None = None
        self._server_thread: threading.Thread | None = None
        self._server_stop_event = threading.Event()

        _require_non_negative("local_dp_rank", self.local_dp_rank)
        _require_non_negative("local_tp_rank", self.local_tp_rank)
        _require_non_negative("listen_port", self.listen_port)
        _require_positive("timeout_s", self.timeout_s)
        _require_positive("max_frame_bytes", self.max_frame_bytes)
        if not self.listen_host:
            raise ValueError("listen_host must be non-empty")
        if self.transport != "zmq":
            raise ValueError("transport must be 'zmq', got "
                             f"{self.transport!r}")

    def start(self) -> None:
        """Start this worker's P-side gather server."""
        with self._lock:
            if self._server_socket is not None:
                return
            socket = self._zmq_context.socket(zmq.ROUTER)
            socket.setsockopt(zmq.LINGER, 0)
            if self.listen_port == 0:
                self.listen_port = int(
                    socket.bind_to_random_port(
                        _zmq_bind_prefix(self.listen_host)))
            else:
                socket.bind(
                    _zmq_tcp_endpoint(self.listen_host, self.listen_port))
            self._server_stop_event.clear()
            self._server_socket = socket
            self._server_thread = threading.Thread(target=self._serve_zmq,
                                                   daemon=True)
            self._server_thread.start()

    def stop(self) -> None:
        """Stop this worker's P-side gather server."""
        with self._lock:
            socket = self._server_socket
            thread = self._server_thread
            self._server_socket = None
            self._server_thread = None
            self._server_stop_event.set()
        if socket is not None:
            socket.close(linger=0)
        if thread is not None:
            thread.join(timeout=self.timeout_s)

    def local_metadata(self) -> RemoteWorkerMetadata:
        """Return metadata that other DP/TP workers can register."""
        return RemoteWorkerMetadata(dp_rank=self.local_dp_rank,
                                    tp_rank=self.local_tp_rank,
                                    worker_id=self.local_worker_id,
                                    tcp_host=self.listen_host,
                                    tcp_port=self.listen_port,
                                    transport=self.transport,
                                    regions=self.local_regions_metadata())

    def local_regions_metadata(self) -> tuple[RegisteredMemoryRegion, ...]:
        """Return locally registered raw regions without TCP endpoint fields."""
        with self._lock:
            return tuple(region.metadata for region in self._local_regions)

    def register_local_region(
        self,
        *,
        buffer: Any,
        region: RegisteredMemoryRegion,
    ) -> RegisteredMemoryRegion:
        """Register one local raw tensor/buffer under an explicit descriptor."""
        metadata = _require_registered_region(region)
        if _is_torch_tensor(buffer):
            _validate_registered_tensor(buffer)
        buffer_size = _buffer_nbytes(buffer)
        if metadata.nbytes > buffer_size:
            raise ValueError("registered region nbytes exceeds buffer size: "
                             f"nbytes={metadata.nbytes} buffer={buffer_size}")
        with self._lock:
            self._check_local_region_id_available(metadata)
            lock = self._local_region_locks_by_buffer.get(id(buffer))
            if lock is None:
                lock = threading.Lock()
                self._local_region_locks_by_buffer[id(buffer)] = lock
            local = _LocalMemoryRegion(metadata=metadata,
                                       buffer=buffer,
                                       write_lock=lock)
            self._local_regions.append(local)
        return metadata

    def _check_local_region_id_available(
            self, metadata: RegisteredMemoryRegion) -> None:
        for region in self._local_regions:
            other = region.metadata
            if metadata.region_id == other.region_id:
                raise ValueError("duplicate memory region id: "
                                 f"{metadata.region_id!r}")

    def register_other_remote_metadata(
            self, metadata: Sequence[RemoteWorkerMetadata]) -> None:
        """Register the peer worker metadata learned during handshake."""
        items = _require_remote_metadata_sequence(metadata)

        with self._lock:
            for remote in items:
                self._remote_metadata[(remote.dp_rank,
                                       remote.tp_rank)] = remote

    def resolve_remote(self, *, tp_rank: int,
                       dp_rank: int) -> RemoteWorkerMetadata:
        """Resolve a registered peer by explicit DP and TP rank."""
        target_dp_rank = int(dp_rank)
        target_tp_rank = int(tp_rank)
        key = (target_dp_rank, target_tp_rank)
        with self._lock:
            try:
                return self._remote_metadata[key]
            except KeyError as exc:
                raise KeyError("remote worker metadata not registered for "
                               f"dp_rank={target_dp_rank} "
                               f"tp_rank={target_tp_rank}") from exc

    def pull_from_registered_into_session(
        self,
        *,
        remote_tp_rank: int,
        dp_rank: int,
        ops: Sequence[StridedTransferOp],
        write_session: "DestinationPageWriteSession",
    ) -> int:
        remote = self.resolve_remote(tp_rank=remote_tp_rank, dp_rank=dp_rank)
        return self.pull_into_session(remote, ops, write_session)

    def pull_into_session(self, remote: RemoteWorkerMetadata,
                          ops: Sequence[StridedTransferOp],
                          write_session: "DestinationPageWriteSession") -> int:
        """Pull compact bytes and scatter into an existing page write session."""
        remote_metadata = _require_remote_metadata(remote)
        op_list = _require_strided_transfer_ops(ops)
        session = _require_destination_page_write_session(write_session)
        compact = self._pull_compact(remote_metadata, op_list)
        return session.scatter(compact, op_list)

    def new_destination_write_session(self) -> "DestinationPageWriteSession":
        """Create a request-scoped D-side page write session."""
        with self._lock:
            if not self._local_regions:
                raise RuntimeError(
                    "strided transfer destination regions are not registered")
            return DestinationPageWriteSession(tuple(self._local_regions))

    def gather_for_request(self, ops: Sequence[StridedTransferOp]) -> bytes:
        """P-side request handler: gather source ranges into compact bytes."""
        with self._lock:
            if not self._local_regions:
                raise RuntimeError(
                    "strided transfer source regions are not registered")
            return gather_strided_ops_from_regions(self._local_regions, ops)

    def _pull_compact(self, remote: RemoteWorkerMetadata,
                      ops: Sequence[StridedTransferOp]) -> bytes:
        if remote.transport != "zmq":
            raise ValueError(f"unsupported transport: {remote.transport}")
        return self._pull_compact_zmq(remote, ops)

    def _request_payload(self,
                         ops: Sequence[StridedTransferOp]) -> dict[str, Any]:
        return {
            "version": _PROTOCOL_VERSION,
            "op": "pull",
            "ops": [op.to_dict() for op in ops],
        }

    def _pull_compact_zmq(self, remote: RemoteWorkerMetadata,
                          ops: Sequence[StridedTransferOp]) -> bytes:
        request = _json_dumps_bytes(self._request_payload(ops))
        _check_frame_size(len(request), self.max_frame_bytes)
        socket = self._zmq_context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        timeout_ms = max(1, int(self.timeout_s * 1000))
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            socket.connect(_zmq_tcp_endpoint(remote.tcp_host, remote.tcp_port))
            socket.send_multipart([_MSG_PULL, request])
            try:
                frames = socket.recv_multipart()
            except zmq.Again as exc:
                raise TimeoutError("timed out waiting for remote strided "
                                   "transfer response") from exc
        finally:
            socket.close(linger=0)
        if not frames:
            raise RuntimeError("empty remote strided transfer response")
        tag = frames[0]
        if tag == _MSG_ERROR:
            if len(frames) != 2:
                raise RuntimeError("malformed remote error response")
            error = _json_loads_bytes(frames[1])
            raise RuntimeError(
                str(error.get("message", "remote transfer failed")))
        if tag != _MSG_OK:
            raise RuntimeError(f"unexpected remote response tag: {tag!r}")
        if len(frames) != 3:
            raise RuntimeError("malformed remote success response")
        response = _json_loads_bytes(frames[1])
        compact = frames[2]
        _check_frame_size(len(compact), self.max_frame_bytes)
        if int(response.get("payload_bytes", -1)) != len(compact):
            raise RuntimeError("remote compact payload size metadata mismatch")
        expected = _total_transfer_bytes(ops)
        if len(compact) != expected:
            raise RuntimeError("remote compact payload size mismatch: got "
                               f"{len(compact)}, expected {expected}")
        return compact

    def _serve_zmq(self) -> None:
        while not self._server_stop_event.is_set():
            socket = self._server_socket
            if socket is None:
                return
            try:
                if socket.poll(timeout=50) == 0:
                    continue
                frames = socket.recv_multipart()
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError:
                if self._server_stop_event.is_set():
                    return
                continue
            response = self._handle_zmq_request(frames)
            if response is None:
                continue
            try:
                socket.send_multipart(response)
            except zmq.ZMQError:
                if self._server_stop_event.is_set():
                    return

    def _handle_zmq_request(self,
                            frames: Sequence[bytes]) -> list[bytes] | None:
        if len(frames) < 2:
            return None
        identity = frames[0]
        try:
            if len(frames) != 3:
                raise ValueError("malformed strided transfer request")
            tag = frames[1]
            if tag != _MSG_PULL:
                raise ValueError(f"unsupported strided transfer tag: {tag!r}")
            request = _json_loads_bytes(frames[2])
            _validate_pull_request(request)
            compact = self.gather_for_request(
                _strided_transfer_ops_from_wire(request["ops"]))
            _check_frame_size(len(compact), self.max_frame_bytes)
            response = _json_dumps_bytes({
                "status": "ok",
                "payload_bytes": len(compact),
            })
            return [identity, _MSG_OK, response, compact]
        except Exception as exc:  # pragma: no cover - exercised by clients.
            response = _json_dumps_bytes({
                "status": "error",
                "message": str(exc),
            })
            return [identity, _MSG_ERROR, response]


class DestinationPageWriteSession:
    """Request-scoped D-side cache for destination region writes."""

    def __init__(self, regions: Sequence[_LocalMemoryRegion]) -> None:
        if isinstance(regions, (str, bytes, bytearray)):
            raise TypeError("regions must be a sequence of local regions")
        self._regions = tuple(regions)
        for region in self._regions:
            if not isinstance(region, _LocalMemoryRegion):
                raise TypeError("regions must contain only local regions")
        self._pages: dict[tuple[int, str, int], _CachedDestinationPage] = {}
        self._tensor_write_patches: dict[int, tuple[_LocalMemoryRegion,
                                                    list[tuple[int,
                                                               bytes]]]] = {}
        self._held_tensor_write_regions: dict[int, _LocalMemoryRegion] = {}

    def scatter(self, compact_buffer: Any,
                ops: Sequence[StridedTransferOp]) -> int:
        op_list = _require_strided_transfer_ops(ops)
        compact = _byte_view(compact_buffer)
        expected_size = _total_transfer_bytes(op_list)
        if len(compact) != expected_size:
            raise ValueError("compact buffer size mismatch: got "
                             f"{len(compact)}, expected {expected_size}")

        compact_offset = 0
        for op in op_list:
            region = _find_region_by_id(self._regions, "destination",
                                        op.dst_region_id)
            for i in range(op.num_segments):
                src_offset = compact_offset + i * op.segment_bytes
                dst_offset = op.dst_offset_bytes + i * op.dst_stride_bytes
                _check_range("compact", src_offset, op.segment_bytes,
                             len(compact))
                payload = compact[src_offset:src_offset + op.segment_bytes]
                if _is_torch_tensor(region.buffer):
                    self._write_to_tensor_cache(region, dst_offset, payload)
                else:
                    self._write_to_region_pages(region, dst_offset, payload)
            compact_offset += op.segment_bytes * op.num_segments
        return expected_size

    def flush_all(self) -> None:
        try:
            for page in tuple(self._pages.values()):
                if page.dirty:
                    _write_region_page(page)
            for region, patches in tuple(self._tensor_write_patches.values()):
                for offset_bytes, payload in sorted(patches):
                    _copy_bytes_to_tensor_range(payload, region.buffer,
                                                offset_bytes)
        finally:
            self._pages.clear()
            self._tensor_write_patches.clear()
            self._release_tensor_write_locks()

    def discard(self) -> None:
        self._pages.clear()
        self._tensor_write_patches.clear()
        self._release_tensor_write_locks()

    def _write_to_tensor_cache(self, region: _LocalMemoryRegion,
                               region_offset: int,
                               payload: memoryview) -> None:
        _check_range("destination", region_offset, len(payload),
                     region.metadata.nbytes)
        itemsize = _tensor_element_size(region.buffer)
        _check_tensor_byte_alignment("destination", region_offset,
                                     len(payload), itemsize)
        patches = self._staged_tensor_writes(region)
        end = region_offset + len(payload)
        for other_offset, other_payload in patches:
            other_end = other_offset + len(other_payload)
            if region_offset < other_end and other_offset < end:
                raise ValueError("overlapping destination tensor ranges: "
                                 f"[{region_offset}, {end}) and "
                                 f"[{other_offset}, {other_end})")
        patches.append((region_offset, bytes(payload)))

    def _staged_tensor_writes(
            self, region: _LocalMemoryRegion) -> list[tuple[int, bytes]]:
        key = id(region.buffer)
        cached = self._tensor_write_patches.get(key)
        if cached is not None:
            return cached[1]
        region.write_lock.acquire()
        self._held_tensor_write_regions[key] = region
        patches: list[tuple[int, bytes]] = []
        self._tensor_write_patches[key] = (region, patches)
        return patches

    def _release_tensor_write_locks(self) -> None:
        for region in reversed(tuple(
                self._held_tensor_write_regions.values())):
            region.write_lock.release()
        self._held_tensor_write_regions.clear()

    def _write_to_region_pages(self, region: _LocalMemoryRegion,
                               region_offset: int,
                               payload: memoryview) -> None:
        _check_range("destination", region_offset, len(payload),
                     region.metadata.nbytes)
        page_bytes = region.metadata.effective_page_bytes
        remaining = len(payload)
        payload_offset = 0
        current_offset = region_offset
        while remaining:
            page_offset = (current_offset // page_bytes) * page_bytes
            page_inner = current_offset - page_offset
            page_length = min(page_bytes, region.metadata.nbytes - page_offset)
            chunk = min(remaining, page_length - page_inner)
            read_existing = not (page_inner == 0 and chunk == page_length)
            page = self._cached_page(region, page_offset, read_existing)
            page.data[page_inner:page_inner +
                      chunk] = payload[payload_offset:payload_offset + chunk]
            page.dirty = True
            remaining -= chunk
            payload_offset += chunk
            current_offset += chunk

    def _cached_page(self, region: _LocalMemoryRegion, page_offset: int,
                     read_existing: bool) -> _CachedDestinationPage:
        key = (id(region.buffer), region.metadata.region_id, page_offset)
        if key not in self._pages:
            page_length = min(region.metadata.effective_page_bytes,
                              region.metadata.nbytes - page_offset)
            if read_existing:
                data = bytearray(
                    _read_bytes(region.buffer, page_offset, page_length))
            else:
                data = bytearray(page_length)
            self._pages[key] = _CachedDestinationPage(region=region,
                                                      page_offset=page_offset,
                                                      data=data)
        return self._pages[key]


__all__ = [
    "DestinationPageWriteSession",
    "RemoteWorkerMetadata",
    "RegisteredMemoryRegion",
    "StridedKVTransferEngine",
    "StridedTransferOp",
]


def gather_strided_ops_from_regions(
    regions: Sequence[_LocalMemoryRegion],
    ops: Sequence[StridedTransferOp],
) -> bytes:
    """P-side gather using region-relative source offsets and page caching."""
    op_list = _require_strided_transfer_ops(ops)
    compact = bytearray(_total_transfer_bytes(op_list))
    compact_offset = 0
    page_cache: dict[tuple[int, int], memoryview] = {}

    for op in op_list:
        for i in range(op.num_segments):
            src_offset = op.src_offset_bytes + i * op.src_stride_bytes
            dst_offset = compact_offset + i * op.segment_bytes
            _copy_from_region_pages(regions, page_cache, op.src_region_id,
                                    src_offset, compact, dst_offset,
                                    op.segment_bytes)
        compact_offset += op.segment_bytes * op.num_segments
    return bytes(compact)


def _require_strided_transfer_ops(
        ops: Sequence[StridedTransferOp]) -> tuple[StridedTransferOp, ...]:
    if isinstance(ops, (str, bytes, bytearray)):
        raise TypeError("ops must be a sequence of StridedTransferOp")
    result = tuple(ops)
    for op in result:
        if not isinstance(op, StridedTransferOp):
            raise TypeError("ops must contain only StridedTransferOp, got "
                            f"{type(op)!r}")
    return result


def _strided_transfer_ops_from_wire(
        ops: Sequence[Mapping[str, Any]]) -> tuple[StridedTransferOp, ...]:
    if isinstance(ops, (str, bytes, bytearray)):
        raise TypeError("wire ops must be a sequence of mappings")
    result = []
    for op in ops:
        if not isinstance(op, Mapping):
            raise TypeError("wire ops must contain only mappings, got "
                            f"{type(op)!r}")
        result.append(StridedTransferOp.from_mapping(op))
    return tuple(result)


def _require_registered_region(
        region: RegisteredMemoryRegion) -> RegisteredMemoryRegion:
    if isinstance(region, RegisteredMemoryRegion):
        return region
    raise TypeError("expected RegisteredMemoryRegion, got "
                    f"{type(region)!r}")


def _require_registered_regions(
    regions: Sequence[RegisteredMemoryRegion]
) -> tuple[RegisteredMemoryRegion, ...]:
    if isinstance(regions, (str, bytes, bytearray)):
        raise TypeError("regions must be a sequence of RegisteredMemoryRegion")
    result = tuple(regions)
    for region in result:
        _require_registered_region(region)
    return result


def _registered_regions_from_wire(
    regions: Sequence[Mapping[str,
                              Any]]) -> tuple[RegisteredMemoryRegion, ...]:
    if isinstance(regions, (str, bytes, bytearray)):
        raise TypeError("wire regions must be a sequence of mappings")
    result = []
    for region in regions:
        if not isinstance(region, Mapping):
            raise TypeError("wire regions must contain only mappings, got "
                            f"{type(region)!r}")
        result.append(RegisteredMemoryRegion.from_mapping(region))
    return tuple(result)


def _require_remote_metadata(
        metadata: RemoteWorkerMetadata) -> RemoteWorkerMetadata:
    if isinstance(metadata, RemoteWorkerMetadata):
        return metadata
    raise TypeError("expected RemoteWorkerMetadata, got "
                    f"{type(metadata)!r}")


def _require_remote_metadata_sequence(
    metadata: Sequence[RemoteWorkerMetadata]
) -> tuple[RemoteWorkerMetadata, ...]:
    if isinstance(metadata, RemoteWorkerMetadata):
        raise TypeError("metadata must be a sequence of RemoteWorkerMetadata")
    if isinstance(metadata, (str, bytes, bytearray)):
        raise TypeError("metadata must be a sequence of RemoteWorkerMetadata")
    result = tuple(metadata)
    for item in result:
        _require_remote_metadata(item)
    return result


def _require_destination_page_write_session(
        write_session: DestinationPageWriteSession
) -> DestinationPageWriteSession:
    if isinstance(write_session, DestinationPageWriteSession):
        return write_session
    raise TypeError("expected DestinationPageWriteSession, got "
                    f"{type(write_session)!r}")


def _total_transfer_bytes(ops: Sequence[StridedTransferOp]) -> int:
    return sum(op.segment_bytes * op.num_segments for op in ops)


def _find_region_by_id(
    regions: Sequence[_LocalMemoryRegion],
    name: str,
    region_id: str,
) -> _LocalMemoryRegion:
    for region in regions:
        if region.metadata.region_id == region_id:
            return region
    raise KeyError(f"{name} memory region is not registered: {region_id!r}")


def _copy_from_region_pages(
    regions: Sequence[_LocalMemoryRegion],
    page_cache: dict[tuple[int, int], memoryview],
    src_region_id: str,
    region_offset: int,
    dst: bytearray,
    dst_offset: int,
    length: int,
) -> None:
    region = _find_region_by_id(regions, "source", src_region_id)
    _check_range("source", region_offset, length, region.metadata.nbytes)
    page_bytes = region.metadata.effective_page_bytes
    page_offset = (region_offset // page_bytes) * page_bytes
    page_inner = region_offset - page_offset
    if page_inner + length > page_bytes:
        raise ValueError("source segment crosses source page boundary: "
                         f"region_id={src_region_id!r} "
                         f"offset={region_offset} length={length} "
                         f"page_bytes={page_bytes}")
    page = _cached_page_view(page_cache, region, page_offset)
    if page_inner + length > len(page):
        raise ValueError("source range cannot be read from cached page")
    dst[dst_offset:dst_offset + length] = page[page_inner:page_inner + length]


def _cached_page_view(
    page_cache: dict[tuple[int, int], memoryview],
    region: _LocalMemoryRegion,
    page_offset: int,
) -> memoryview:
    key = (id(region.buffer), page_offset)
    if key not in page_cache:
        page_bytes = min(region.metadata.effective_page_bytes,
                         region.metadata.nbytes - page_offset)
        page_cache[key] = memoryview(
            _read_bytes(region.buffer, page_offset, page_bytes))
    return page_cache[key]


def _write_region_page(page: _CachedDestinationPage) -> None:
    _write_bytes(page.region.buffer, page.page_offset, page.data)


def _buffer_nbytes(buffer: Any) -> int:
    if hasattr(buffer, "nbytes"):
        return int(buffer.nbytes)
    if hasattr(buffer, "read_bytes"):
        raise ValueError(
            "buffer with read_bytes must also expose an nbytes attribute")
    if _is_torch_tensor(buffer):
        return int(buffer.numel() * buffer.element_size())
    return len(_byte_view(buffer))


def _read_bytes(buffer: Any, offset: int, length: int) -> bytes:
    if hasattr(buffer, "read_bytes"):
        return bytes(buffer.read_bytes(offset, length))
    if _is_torch_tensor(buffer):
        return _tensor_slice_to_bytes(buffer, offset, length)
    view = _byte_view(buffer)
    _check_range("source", offset, length, len(view))
    return bytes(view[offset:offset + length])


def _write_bytes(buffer: Any, offset: int,
                 data: bytes | bytearray | memoryview) -> None:
    payload = _byte_view(data)
    if hasattr(buffer, "write_bytes"):
        buffer.write_bytes(offset, payload)
        return
    view = _byte_view(buffer)
    if view.readonly:
        raise ValueError("destination buffer must be writable")
    _check_range("destination", offset, len(payload), len(view))
    view[offset:offset + len(payload)] = payload


def _byte_view(buffer: Any) -> memoryview:
    if _is_torch_tensor(buffer):
        return memoryview(_tensor_to_bytes(buffer))
    view = memoryview(buffer)
    if view.format != "B" or view.itemsize != 1 or view.ndim != 1:
        view = view.cast("B")
    return view


def _is_torch_tensor(buffer: Any) -> bool:
    torch = _optional_torch()
    return torch is not None and isinstance(buffer, torch.Tensor)


def _optional_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _require_torch() -> Any:
    torch = _optional_torch()
    if torch is None:
        raise TypeError("torch tensor buffers require torch to be installed")
    return torch


def _validate_registered_tensor(tensor: Any) -> None:
    torch = _require_torch()
    supported_dtypes = {
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bfloat16,
    }
    if tensor.dtype not in supported_dtypes:
        raise TypeError("ZMQ tensor regions support only FP8/BF16, got "
                        f"{tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError("ZMQ tensor regions must be contiguous")


def _tensor_to_bytes(tensor: Any) -> bytes:
    torch = _require_torch()
    if not tensor.is_contiguous():
        raise ValueError("torch tensor byte buffers must be contiguous")
    cpu_tensor = tensor.detach().cpu().contiguous()
    return cpu_tensor.view(torch.uint8).numpy().tobytes()


def _tensor_slice_to_bytes(tensor: Any, offset: int, length: int) -> bytes:
    torch = _require_torch()
    if not tensor.is_contiguous():
        raise ValueError("torch tensor byte buffers must be contiguous")
    itemsize = _tensor_element_size(tensor)
    _check_range("source", offset, length,
                 int(tensor.numel() * tensor.element_size()))
    _check_tensor_byte_alignment("source", offset, length, itemsize)
    element_offset = int(offset) // itemsize
    element_count = int(length) // itemsize
    typed_tensor = tensor.detach().reshape(-1).narrow(0, element_offset,
                                                      element_count)
    return typed_tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _copy_bytes_to_tensor_range(
    payload: bytes | bytearray | memoryview,
    tensor: Any,
    offset_bytes: int,
) -> None:
    torch = _require_torch()
    if not tensor.is_contiguous():
        raise ValueError("torch tensor byte buffers must be contiguous")
    itemsize = _tensor_element_size(tensor)
    payload_bytes = bytearray(payload)
    _check_tensor_byte_alignment("destination", offset_bytes,
                                 len(payload_bytes), itemsize)
    _check_range("destination", offset_bytes, len(payload_bytes),
                 tensor.numel() * itemsize)
    cpu_bytes = torch.frombuffer(payload_bytes, dtype=torch.uint8)
    cpu_typed = cpu_bytes.view(tensor.dtype)
    element_offset = offset_bytes // itemsize
    target = tensor.reshape(-1).narrow(0, element_offset, cpu_typed.numel())
    target.copy_(cpu_typed.to(device=tensor.device))


def _tensor_element_size(tensor: Any) -> int:
    itemsize = int(tensor.element_size())
    _require_positive("tensor element_size", itemsize)
    return itemsize


def _check_tensor_byte_alignment(name: str, offset: int, length: int,
                                 itemsize: int) -> None:
    if offset % itemsize != 0 or length % itemsize != 0:
        raise ValueError(f"{name} tensor byte range must be aligned to "
                         f"tensor element_size={itemsize}: "
                         f"offset={offset} length={length}")


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _check_range(name: str, offset: int, size: int, total: int) -> None:
    if offset < 0 or size < 0 or offset > total or size > total - offset:
        raise ValueError(
            f"{name} range out of bounds: offset={offset} size={size} "
            f"total={total}")


def _check_frame_size(size: int, max_frame_bytes: int) -> None:
    if size < 0 or size > max_frame_bytes:
        raise ValueError(
            f"frame size out of bounds: size={size} max={max_frame_bytes}")


def _zmq_tcp_endpoint(host: str, port: int) -> str:
    if not host:
        raise ValueError("host must be non-empty")
    return f"tcp://{host}:{int(port)}"


def _zmq_bind_prefix(host: str) -> str:
    if not host:
        raise ValueError("host must be non-empty")
    return f"tcp://{host}"


def _json_dumps_bytes(data: Mapping[str, Any]) -> bytes:
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def _json_loads_bytes(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON payload must be an object")
    return value


def _validate_pull_request(request: Mapping[str, Any]) -> None:
    if int(request.get("version", -1)) != _PROTOCOL_VERSION:
        raise ValueError("unsupported strided transfer protocol version")
    if request.get("op") != "pull":
        raise ValueError("unsupported strided transfer op")
    if not isinstance(request.get("ops"), list):
        raise ValueError("pull request must contain an ops list")
