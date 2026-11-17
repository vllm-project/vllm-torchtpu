# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hand-off of activations between two TPU slices, over TCP.

Stages inside a slice hand activations on over ICI (``pp_buffered``). Two
slices share no ICI, so the hop from the last stage of one slice to the
first stage of the next goes over the host network. A ``DcnLink`` is one
direction of that hop, between the devices that hold the same column in the
two stages. The sending end starts copying a forward's outputs to host
buffers as soon as the forward is submitted (``stage``) and writes them to
the receiving host once the forward's fence shows them copied (``send``);
the receiving end reads them into a host buffer of its own and uploads them
to a fresh device tensor.

A copy to the host completes only after everything the device was given
before it. Started when the send thread takes a transfer up, it would also
wait for the forwards queued behind the one that produced it; started with
the forward, it runs right behind it. Several transfers are staged before
the first of them is written, so the sending end keeps a free list of host
buffers per carried tensor and row count and takes a buffer back once its
stripes are written. The receiving end starts each upload without waiting
for it: the device runs a reader of the tensor after the upload, but the
upload reads its host buffer until it is done. So the receiving end keeps a
free list too, takes a transfer's buffers back once the forward that read
it has completed (``landed``), and holds as many buffers per carried tensor
and row count as receives run ahead of completed forwards.

A link holds ``TPU_PP_DCN_STREAMS`` TCP connections, each with a thread of
its own. A tensor travels as that many contiguous stripes, one per
connection, written and read in parallel: one connection alone does not
fill a host's network. A fixed header precedes every stripe and names its
transfer, tensor, row count, the tensor's size and the stripe's place in
it. The receiver checks each header against the receive it is serving, so
ends that disagree fail at once instead of handing a forward the wrong
data.

The receiving end listens on its host's address and publishes it in the
start-up rendezvous store under its engine rank; the sending end waits for
that key and connects. Neither waits for the other while it is built: the
receiver accepts on its first receive, the sender connects on its first
send. A send returns once its stripes are written, and the kernel's socket
buffers hold what the receiver has not read yet; once they are full, the
send waits for the receiver to read, which is the hop's back-pressure. A
receive returns once the data has arrived and its upload has started.

A failure on any connection fails the link: it shuts every connection down,
so the other end fails too, and this call and every later one raise. There
are no retries.
"""

import contextlib
import errno
import queue
import socket
import struct
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, Future, wait
from functools import partial

import torch
from torch.distributed import Store
from torch.distributed.constants import default_pg_timeout
from vllm.utils.network_utils import (
    get_ip,
    is_valid_ipv6_address,
    join_host_port,
    split_host_port,
)
from vllm.utils.torch_utils import PIN_MEMORY

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.tracing.annotation import TraceAnnotation

logger = init_logger(__name__)

# The receiving end's "host:port", keyed by its engine rank.
_ADDRESS_KEY = "tpu_pp_dcn_link_{}"

# Sent once on each connection: which stripe of every tensor it carries.
_STREAM = struct.Struct("!q")

# Precedes every stripe: transfer, tensor index, rows, the tensor's bytes,
# the stripe's offset and its bytes.
_HEADER = struct.Struct("!6q")

# numpy has no bfloat16, so a tensor's bytes are reached through an integer
# view of the same item size.
_INTEGER_OF_SIZE = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _start_to_host(tensor: torch.Tensor, host: torch.Tensor) -> None:
    """Starts copying a device tensor into a host buffer of its shape and
    dtype. The copy completes after everything the device was given before
    it, so a fence recorded after it shows it done."""
    host.copy_(tensor, non_blocking=True)


def _start_to_device(host: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Starts uploading a host buffer to a fresh device tensor and returns the
    tensor. The device runs a reader of the tensor after the upload, but the
    upload reads the buffer until it is done, so the buffer takes no other
    transfer before the reader has completed (``DcnLink.landed``)."""
    return host.to(device, non_blocking=True, copy=True)


def _byte_view(host: torch.Tensor) -> memoryview:
    """The bytes of a contiguous host tensor, writable in place."""
    flat = host.view(_INTEGER_OF_SIZE[host.element_size()]).reshape(-1)
    return memoryview(flat.numpy()).cast("B")


def _read_into(sock: socket.socket, view: memoryview) -> None:
    """Fills ``view`` from ``sock``."""
    while len(view):
        count = sock.recv_into(view)
        if count == 0:
            raise ConnectionError("the other end closed the connection")
        view = view[count:]


def _write_stripe(
    sock: socket.socket, header: tuple[int, ...], data: memoryview
) -> None:
    sock.sendall(_HEADER.pack(*header))
    sock.sendall(data)


def _describe(header: tuple[int, ...]) -> str:
    transfer, index, rows, total, offset, size = header
    return (
        f"transfer {transfer}, tensor {index} of {rows} rows ({total} bytes), "
        f"{size} bytes at {offset}"
    )


def _read_stripe(
    sock: socket.socket, expected: tuple[int, ...], data: memoryview
) -> None:
    header = bytearray(_HEADER.size)
    _read_into(sock, memoryview(header))
    sent = _HEADER.unpack(header)
    if sent != expected:
        raise RuntimeError(
            f"this end expects {_describe(expected)}; the sending end sent "
            f"{_describe(sent)}"
        )
    _read_into(sock, data)


# Writes or reads one stripe: the connection, the stripe's header, its bytes.
_StripeMove = Callable[[socket.socket, tuple[int, ...], memoryview], None]


class _Stream:
    """One connection of a link and the thread that moves its stripes. The
    thread is a daemon: a process that exits must not wait for a peer that
    holds the connection open."""

    def __init__(self, sock: socket.socket, name: str):
        self.sock = sock
        self._jobs: queue.SimpleQueue[
            tuple[Callable[[], None], Future[None]] | None
        ] = queue.SimpleQueue()
        # Orders a job against close: a job is queued ahead of the thread's
        # stop, or refused.
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def submit(self, job: Callable[[], None]) -> Future[None]:
        """Queues a job for the thread; a closed stream fails it at once, so
        its caller never waits for a thread that has stopped."""
        future: Future[None] = Future()
        with self._lock:
            if self._closed:
                future.set_exception(ConnectionError("the connection was closed"))
            else:
                self._jobs.put((job, future))
        return future

    def _loop(self) -> None:
        while (item := self._jobs.get()) is not None:
            job, future = item
            try:
                job()
            except BaseException as exc:  # re-raised on the caller's thread
                future.set_exception(exc)
            else:
                future.set_result(None)

    def shut_down(self) -> None:
        """Ends the connection both ways, which wakes a read or write blocked
        on it."""
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError as exc:
            # The other end has disconnected already, or the link has closed
            # the socket.
            if exc.errno not in (errno.ENOTCONN, errno.EBADF):
                raise

    def close(self) -> None:
        self.shut_down()
        with self._lock:
            if not self._closed:
                self._closed = True
                self._jobs.put(None)
        self._thread.join()
        self.sock.close()


class DcnLink:
    """One direction of a hand-off between the same column of two slices.

    Both ends are built with the sending and the receiving engine rank,
    ``src`` and ``dst``; ``rank`` says which end this is. ``specs`` holds the
    row shape and dtype of each carried tensor, in the order they travel.
    """

    def __init__(
        self,
        store: Store,
        src: int,
        dst: int,
        rank: int,
        specs: tuple[tuple[tuple[int, ...], torch.dtype], ...],
        device: torch.device,
    ):
        streams = envs.TPU_PP_DCN_STREAMS
        if streams < 1:
            raise ValueError(f"TPU_PP_DCN_STREAMS must be at least 1, got {streams}.")
        if rank not in (src, dst):
            raise ValueError(
                f"engine rank {rank} is neither end of the link {src} -> {dst}"
            )
        self.name = f"PP DCN link {src} -> {dst}"
        self.store = store
        self.src = src
        self.dst = dst
        self.specs = specs
        self.device = device
        self.streams = streams
        self._receiving = rank == dst
        self._key = _ADDRESS_KEY.format(dst)
        self._listener: socket.socket | None = None
        self._connections: list[_Stream] = []
        # Free host buffers by (tensor index, rows). On the sending end stage
        # takes them on the thread that submits forwards and send gives them
        # back on the send thread; on the receiving end receive takes them on
        # the thread that posts receives and landed gives them back on the
        # send thread.
        self._free: dict[tuple[int, int], list[torch.Tensor]] = {}
        # The receiving end's transfers whose forwards have not completed, in
        # transfer order, with the host buffers their uploads read.
        self._landing: deque[tuple[int, tuple[torch.Tensor, ...]]] = deque()
        # The device tensor each staged buffer is being filled from, by the
        # buffer's id, kept alive until the buffer is written.
        self._sources: dict[int, torch.Tensor] = {}
        self._free_lock = threading.Lock()
        self._error: BaseException | None = None
        if self._receiving:
            ip = get_ip()
            family = socket.AF_INET6 if is_valid_ipv6_address(ip) else socket.AF_INET
            self._listener = socket.create_server(
                (ip, 0), family=family, backlog=streams
            )
            # The sending end connects on its first send, once its model has
            # loaded, which can be minutes after this end; the wait is
            # bounded like a start-up collective.
            self._listener.settimeout(default_pg_timeout.total_seconds())
            port = self._listener.getsockname()[1]
            store.set(self._key, join_host_port(ip, port))

    def stage(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Starts copying a forward's outputs to host buffers, right behind
        the forward on the device, and returns the buffers for ``send``."""
        try:
            self._raise_failure()
            if len(tensors) != len(self.specs):
                raise RuntimeError(
                    f"{self.name}: a forward hands on {len(tensors)} tensors; "
                    f"the link carries {len(self.specs)}"
                )
            for index, (tensor, (shape, dtype)) in enumerate(zip(tensors, self.specs)):
                if tuple(tensor.shape[1:]) != tuple(shape) or tensor.dtype != dtype:
                    raise RuntimeError(
                        f"{self.name}: tensor {index} is "
                        f"{tuple(tensor.shape)} {tensor.dtype}; the link carries "
                        f"[rows, {', '.join(map(str, shape))}] {dtype}"
                    )
            staged = []
            with TraceAnnotation("PP:DcnToHost"):
                for index, tensor in enumerate(tensors):
                    host = self._take(index, tensor)
                    _start_to_host(tensor, host)
                    staged.append(host)
            return tuple(staged)
        except BaseException as exc:
            self._fail(exc)
            raise

    def send(self, transfer: int, staged: tuple[torch.Tensor, ...]) -> None:
        """Writes one transfer's staged host buffers to the receiving end and
        returns when every stripe is written. The copies into them must be
        done, as a fence recorded after ``stage`` shows; each buffer goes back
        to the link once its stripes are written."""
        try:
            self._open()
            with TraceAnnotation("PP:DcnWire"):
                for index, host in enumerate(staged):
                    rows = int(host.shape[0])
                    self._move(_write_stripe, transfer, index, rows, _byte_view(host))
                    self._give_back(index, host)
        except BaseException as exc:
            self._fail(exc)
            raise

    def receive(self, transfer: int, rows: int) -> tuple[torch.Tensor, ...]:
        """Reads one transfer of ``rows`` rows from the sending end into host
        buffers and returns its tensors once the data has arrived and their
        uploads have started; the device runs a reader of a tensor after its
        upload. The buffers stay with the transfer until ``landed``."""
        try:
            self._open()
            hosts = []
            received = []
            for index in range(len(self.specs)):
                host = self._free_host(index, rows)
                hosts.append(host)
                with TraceAnnotation("PP:DcnWire"):
                    self._move(_read_stripe, transfer, index, rows, _byte_view(host))
                with TraceAnnotation("PP:DcnToDevice"):
                    received.append(_start_to_device(host, self.device))
            with self._free_lock:
                self._landing.append((transfer, tuple(hosts)))
            return tuple(received)
        except BaseException as exc:
            self._fail(exc)
            raise

    def landed(self, transfer: int) -> None:
        """Takes back the host buffers of every transfer up to ``transfer``:
        the forwards that read them have completed on the device, so their
        uploads are done."""
        with self._free_lock:
            done = []
            while self._landing and self._landing[0][0] <= transfer:
                done.append(self._landing.popleft()[1])
        for hosts in done:
            for index, host in enumerate(hosts):
                self._give_back(index, host)

    def close(self) -> None:
        """Ends the connections and stops their threads; for tests and
        shutdown. Another thread may be in a receive that waits for the
        sending end or for its data: that receive raises. A closed link
        refuses every later call."""
        if self._error is None:
            self._error = ConnectionError(f"{self.name} was closed")
        listener = self._listener
        if listener is not None:
            # Wakes an accept blocked on it.
            with contextlib.suppress(OSError):
                listener.shutdown(socket.SHUT_RDWR)
            listener.close()
            self._listener = None
        for connection in self._connections:
            connection.close()
        self._connections = []

    def _new_host(self, index: int, rows: int) -> torch.Tensor:
        shape, dtype = self.specs[index]
        return torch.empty((rows, *shape), dtype=dtype, pin_memory=PIN_MEMORY)

    def _free_host(self, index: int, rows: int) -> torch.Tensor:
        """A free host buffer for tensor ``index`` of ``rows`` rows."""
        with self._free_lock:
            free = self._free.get((index, rows))
            host = free.pop() if free else None
        if host is None:
            host = self._new_host(index, rows)
        return host

    def _take(self, index: int, tensor: torch.Tensor) -> torch.Tensor:
        """A free host buffer of the sending end for ``tensor``, the carried
        tensor ``index``; the link holds ``tensor`` until the buffer comes
        back, so the device keeps its data while the copy is pending."""
        host = self._free_host(index, int(tensor.shape[0]))
        with self._free_lock:
            self._sources[id(host)] = tensor
        return host

    def _give_back(self, index: int, host: torch.Tensor) -> None:
        with self._free_lock:
            self._sources.pop(id(host), None)
            self._free.setdefault((index, int(host.shape[0])), []).append(host)

    def _raise_failure(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"{self.name} has failed") from self._error

    def _open(self) -> None:
        """Opens the link's connections on first use."""
        self._raise_failure()
        if not self._connections:
            socks = self._accept() if self._receiving else self._connect()
            self._connections = [
                _Stream(sock, f"pp-dcn-{self.src}-{self.dst}-{stream}")
                for stream, sock in enumerate(socks)
            ]
            if self._error is not None:
                # The link closed or failed while the connections opened.
                for connection in self._connections:
                    connection.close()
                self._connections = []
                self._raise_failure()
            logger.info(
                "%s: %d TCP streams %s %s",
                self.name,
                self.streams,
                "from" if self._receiving else "to",
                socks[0].getpeername()[0],
            )

    def _accept(self) -> list[socket.socket]:
        """Takes the sending end's connections, in stream order; closes the
        ones it took when it fails, as when close() wakes it."""
        listener = self._listener
        if listener is None:
            raise RuntimeError(f"{self.name} was closed") from self._error
        taken: list[socket.socket] = []
        socks: dict[int, socket.socket] = {}
        try:
            while len(socks) < self.streams:
                sock, _ = listener.accept()
                taken.append(sock)
                announced = bytearray(_STREAM.size)
                _read_into(sock, memoryview(announced))
                (stream,) = _STREAM.unpack(announced)
                if not 0 <= stream < self.streams or stream in socks:
                    raise RuntimeError(
                        f"{self.name}: a connection announced stream {stream}; "
                        f"this end takes streams 0 to {self.streams - 1} once each"
                    )
                socks[stream] = sock
        except BaseException:
            for sock in taken:
                sock.close()
            raise
        finally:
            listener.close()
            self._listener = None
        return [socks[stream] for stream in range(self.streams)]

    def _connect(self) -> list[socket.socket]:
        """Opens the connections once the receiving end has published its
        address; it does so when it is built."""
        self.store.wait([self._key], default_pg_timeout)
        host, port = split_host_port(self.store.get(self._key).decode())
        socks = []
        for stream in range(self.streams):
            sock = socket.create_connection((host, port))
            # Nagle's algorithm would hold a stripe's last partial segment
            # until the receiver acknowledges the one before it.
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(_STREAM.pack(stream))
            socks.append(sock)
        return socks

    def _move(
        self,
        stripe: _StripeMove,
        transfer: int,
        index: int,
        rows: int,
        data: memoryview,
    ) -> None:
        """Moves one tensor's bytes, ``data``, as one stripe per connection,
        all at once: ``stripe`` writes or reads one. Returns when every stripe
        is done; raises the first failure."""
        connections = self._connections
        if len(connections) != self.streams:
            # close() took them while this transfer was under way.
            self._raise_failure()
            raise RuntimeError(f"{self.name} has no open connections")
        total = len(data)
        bounds = [total * stream // self.streams for stream in range(self.streams + 1)]
        futures = [
            connection.submit(
                partial(
                    stripe,
                    connection.sock,
                    (transfer, index, rows, total, start, stop - start),
                    data[start:stop],
                )
            )
            for connection, start, stop in zip(connections, bounds, bounds[1:])
        ]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for stream, future in enumerate(futures):
            if future in done and (error := future.exception()) is not None:
                # The other jobs may wait for an end that has stopped reading
                # or writing; ending the connections returns them.
                self._shut_down()
                wait(futures)
                raise RuntimeError(f"{self.name}, stream {stream}: {error}") from error

    def _shut_down(self) -> None:
        for connection in self._connections:
            connection.shut_down()

    def _fail(self, error: BaseException) -> None:
        """Fails the link and ends its connections, so the other end fails
        too."""
        if self._error is None:
            self._error = error
        self._shut_down()
