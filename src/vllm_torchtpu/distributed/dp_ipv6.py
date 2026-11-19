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
"""Data parallelism on an IPv6-only host.

vLLM's DP rank 0 binds the stateless DP group's listen socket with
``socket.AF_INET`` (``ParallelConfig._pick_stateless_dp_port``), so a
``--data-parallel-address`` such as ``::1`` fails every engine core with
``socket.gaierror: [Errno -9] Address family for hostname not supported``.
This patch binds the same socket with ``AF_INET6`` when the DP master address
is an IPv6 literal. Any other address keeps vLLM's own method unchanged.
"""

import ipaddress
import socket

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _is_ipv6(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.strip("[]")).version == 6
    except ValueError:
        return False


def patch_stateless_dp_port_ipv6() -> None:
    from vllm.config import ParallelConfig

    if getattr(ParallelConfig, "_tpu_dp_ipv6_patch", False):
        return
    original = ParallelConfig._pick_stateless_dp_port

    def _pick_stateless_dp_port(self):
        host = self.data_parallel_master_ip
        if not _is_ipv6(host) or not getattr(self, "_coord_store_port", None):
            return original(self)
        from vllm.distributed.utils import get_cached_tcp_store_client

        store = get_cached_tcp_store_client(host, self._coord_store_port)
        key = "dp_master_port"
        if self.data_parallel_rank != 0:
            return int(store.get(key).decode()), None
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.bind((host.strip("[]"), 0))
        s.listen()
        port = s.getsockname()[1]
        store.set(key, str(port).encode())
        logger.info_once("DP group listen socket bound on IPv6 %s:%d.", host, port)
        return port, s

    ParallelConfig._pick_stateless_dp_port = _pick_stateless_dp_port
    ParallelConfig._tpu_dp_ipv6_patch = True
