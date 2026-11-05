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
"""Tests for waiting on the world TCPStore's port before connecting."""

import importlib
import multiprocessing
import socket
import threading
import time
from datetime import timedelta

import torch.distributed as dist

from vllm_torchtpu.distributed.tcp_store_wait import (
    wait_for_port,
    wait_for_store_port,
)

HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def test_wait_for_port_returns_once_the_port_listens():
    port = _free_port()
    listener = socket.socket()

    def listen_late():
        time.sleep(0.6)
        listener.bind((HOST, port))
        listener.listen()

    thread = threading.Thread(target=listen_late)
    thread.start()
    try:
        waited = wait_for_port(HOST, port, timeout_s=10.0)
    finally:
        thread.join()
        listener.close()
    assert waited is not None and 0.5 <= waited < 1.5


def test_wait_for_port_gives_up_at_the_timeout():
    started = time.monotonic()
    assert wait_for_port(HOST, _free_port(), timeout_s=0.5) is None
    assert time.monotonic() - started < 1.5


def test_the_store_host_does_not_wait():
    rendezvous = importlib.import_module("torch.distributed.rendezvous")
    started = time.monotonic()
    with wait_for_store_port():
        store = rendezvous._create_c10d_store(
            HOST, _free_port(), 0, 1, timedelta(seconds=30)
        )
    assert time.monotonic() - started < 2.0
    store.set("key", "value")
    assert store.get("key") == b"value"


def _client(port: int, rank: int, world_size: int, ready) -> None:
    rendezvous = importlib.import_module("torch.distributed.rendezvous")
    ready.put(rank)
    with wait_for_store_port():
        rendezvous._create_c10d_store(
            HOST, port, rank, world_size, timedelta(seconds=60)
        )


def test_clients_started_early_join_a_late_server_at_once():
    """Clients that start connecting well before the server (the headless
    hosts' workers) would otherwise reconnect seconds after it listens, and
    the server, which waits for every client, waits that long."""
    clients, lead_s = 4, 8.0
    port = _free_port()
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    processes = [
        context.Process(target=_client, args=(port, rank, clients + 1, ready))
        for rank in range(1, clients + 1)
    ]
    for process in processes:
        process.start()
    try:
        for _ in processes:
            ready.get(timeout=60)
        time.sleep(lead_s)
        started = time.monotonic()
        dist.TCPStore(
            HOST,
            port,
            world_size=clients + 1,
            is_master=True,
            wait_for_workers=True,
            timeout=timedelta(seconds=60),
        )
        waited = time.monotonic() - started
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
    assert waited < 1.5, f"the server waited {waited:.1f} s for its clients"
