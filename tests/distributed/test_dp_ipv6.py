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
import socket
from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed import dp_ipv6


def test_is_ipv6():
    assert dp_ipv6._is_ipv6("::1")
    assert dp_ipv6._is_ipv6("[::1]")
    assert not dp_ipv6._is_ipv6("127.0.0.1")
    assert not dp_ipv6._is_ipv6("localhost")


class _Store:
    def __init__(self):
        self.kv = {}

    def set(self, k, v):
        self.kv[k] = v

    def get(self, k):
        return self.kv[k]


@pytest.mark.skipif(not socket.has_ipv6, reason="no IPv6")
def test_rank0_binds_ipv6_and_ipv4_keeps_original(monkeypatch):
    from vllm.config import ParallelConfig

    calls = []
    monkeypatch.setattr(
        ParallelConfig, "_pick_stateless_dp_port", lambda self: calls.append(1) or (7, None)
    )
    monkeypatch.delattr(ParallelConfig, "_tpu_dp_ipv6_patch", raising=False)
    store = _Store()
    import vllm.distributed.utils as vdu

    monkeypatch.setattr(vdu, "get_cached_tcp_store_client", lambda h, p: store)
    dp_ipv6.patch_stateless_dp_port_ipv6()
    pick = ParallelConfig._pick_stateless_dp_port
    v4 = SimpleNamespace(data_parallel_master_ip="127.0.0.1", _coord_store_port=1)
    assert pick(v4) == (7, None) and calls == [1]
    try:
        socket.socket(socket.AF_INET6).bind(("::1", 0))
    except OSError:
        pytest.skip("loopback ::1 unavailable")
    r0 = SimpleNamespace(data_parallel_master_ip="::1", _coord_store_port=1, data_parallel_rank=0)
    port, s = pick(r0)
    assert s.family == socket.AF_INET6 and port == s.getsockname()[1]
    r1 = SimpleNamespace(data_parallel_master_ip="::1", _coord_store_port=1, data_parallel_rank=1)
    assert pick(r1) == (port, None)
    s.close()
