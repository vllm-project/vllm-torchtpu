# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

PULL_START = b"START"
PULL_END = b"END"
MSG_OK = b"OK"
MSG_ERR = b"ERR"

LifecycleHandler = Callable[[int], tuple[bool, str]]


def send_request(
    *,
    host: str,
    port: int,
    tag: bytes,
    uuid: int,
    timeout_s: float,
    action: str,
) -> list[bytes]:
    import zmq
    from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

    sock_path = make_zmq_path("tcp", host, port)
    ctx = zmq.Context.instance()
    sock = make_zmq_socket(ctx=ctx,
                           path=sock_path,
                           socket_type=zmq.DEALER,
                           bind=False)
    timeout_ms = max(1, int(float(timeout_s) * 1000))
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        sock.send_multipart([tag, str(int(uuid)).encode("utf-8")])
        try:
            return sock.recv_multipart()
        except zmq.Again as exc:
            raise TimeoutError(
                f"timed out waiting for producer strided {action} response "
                f"uuid={uuid}") from exc
    finally:
        sock.close(linger=0)


def serve_lifecycle_requests(
    *,
    node_id: int,
    zmq_context: Any,
    side_channel_port: int,
    stop_event: Any,
    handle_pull_start: LifecycleHandler,
    handle_pull_end: LifecycleHandler,
    log: logging.Logger,
) -> None:
    import zmq
    from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

    sock_path = make_zmq_path("tcp", "*", side_channel_port)
    sock = make_zmq_socket(ctx=zmq_context,
                           path=sock_path,
                           socket_type=zmq.ROUTER,
                           bind=True)
    log.info("TPUConnectorV2Worker(%d) rank0 --> side channel on %s", node_id,
             sock_path)
    while not stop_event.is_set():
        try:
            if sock.poll(timeout=500) == 0:
                continue
            frames = sock.recv_multipart()
        except zmq.ContextTerminated:
            return
        except zmq.ZMQError as exc:
            log.warning(
                "TPUConnectorV2Worker(%d) rank0 --> notif recv error: %s",
                node_id, exc)
            continue
        if len(frames) != 3:
            log.warning(
                "TPUConnectorV2Worker(%d) rank0 --> malformed notif frames=%d",
                node_id, len(frames))
            continue
        client_id, tag, uuid_bytes = frames
        if tag == PULL_START:
            _reply_lifecycle_result(sock, client_id, uuid_bytes,
                                    handle_pull_start)
        elif tag == PULL_END:
            _reply_lifecycle_result(sock, client_id, uuid_bytes,
                                    handle_pull_end)
        else:
            log.warning(
                "TPUConnectorV2Worker(%d) rank0 --> unknown notif tag=%s",
                node_id, tag)


def _reply_lifecycle_result(
    sock: Any,
    client_id: bytes,
    uuid_bytes: bytes,
    handler: LifecycleHandler,
) -> None:
    try:
        uuid = int(uuid_bytes.decode("utf-8"))
    except ValueError:
        sock.send_multipart([client_id, MSG_ERR, b"invalid uuid"])
        return
    ok, message = handler(uuid)
    if ok:
        sock.send_multipart([client_id, MSG_OK])
    else:
        sock.send_multipart([client_id, MSG_ERR, message.encode("utf-8")])
