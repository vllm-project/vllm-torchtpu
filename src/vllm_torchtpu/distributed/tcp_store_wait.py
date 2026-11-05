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
"""Wait for the world TCPStore's port before connecting to it.

On a multi-host launch the headless hosts' workers reach
``init_process_group`` long before the head host's rank 0 starts the world
TCPStore server: the head first starts the API server and the EngineCore.
torch's store client retries a refused connect with exponential backoff and
jitter (up to 60 s between tries), so after tens of seconds of refusals the
clients reconnect up to ~25 s after the server listens, and rank 0, whose
store waits for every client, waits that long. Inside
``wait_for_store_port()`` a ``tcp://`` rendezvous client first polls the
store's port with plain sockets every 0.2 s, so it connects at its first try
once the server listens.
"""

import contextlib
import importlib
import socket
import time
from collections.abc import Generator
from datetime import timedelta

from torch.distributed.constants import default_pg_timeout

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

POLL_S = 0.2


def wait_for_port(host: str, port: int, timeout_s: float) -> float | None:
    """Seconds until ``host:port`` accepted a TCP connection, or None if it
    did not within ``timeout_s``."""
    started = time.monotonic()
    deadline = started + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        try:
            with socket.create_connection(
                (host, port), timeout=max(0.05, min(1.0, remaining))
            ):
                return time.monotonic() - started
        except OSError:
            if time.monotonic() + POLL_S > deadline:
                return None
            time.sleep(POLL_S)


@contextlib.contextmanager
def wait_for_store_port() -> Generator[None, None, None]:
    """Make ``tcp://`` rendezvous store clients wait for the store's port
    (``torch.distributed.rendezvous._create_c10d_store``); the store host,
    rank 0 without torchelastic's agent store, does not wait."""
    rendezvous = importlib.import_module("torch.distributed.rendezvous")
    original = rendezvous._create_c10d_store

    def create(hostname, port, rank, world_size, timeout, *args, **kwargs):
        if rank != 0 or rendezvous._torchelastic_use_agent_store():
            limit = timeout if isinstance(timeout, timedelta) else default_pg_timeout
            waited = wait_for_port(hostname, port, limit.total_seconds())
            if waited is None:
                logger.warning(
                    "TCPStore at %s:%d did not accept within %s (rank %d)",
                    hostname,
                    port,
                    limit,
                    rank,
                )
            else:
                logger.info(
                    "TCPStore at %s:%d accepted after %.1f s (rank %d)",
                    hostname,
                    port,
                    waited,
                    rank,
                )
        return original(hostname, port, rank, world_size, timeout, *args, **kwargs)

    setattr(rendezvous, "_create_c10d_store", create)  # noqa: B010
    try:
        yield
    finally:
        setattr(rendezvous, "_create_c10d_store", original)  # noqa: B010
