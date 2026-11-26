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
"""Unit tests for self-published IPv6 slicebuilder host resolution."""

import socket
import subprocess
import sys

import pytest
import ray.cloudpickle as cloudpickle

from vllm_torchtpu.distributed import slicebuilder_hosts as sb


@pytest.fixture(autouse=True)
def _clean_lws_env(monkeypatch):
    for key in ("LWS_LEADER_ADDRESS", "LWS_WORKER_INDEX"):
        monkeypatch.delenv(key, raising=False)


def test_normalize_ip_and_lws_pod_hostname():
    assert sb.normalize_ip("[2001:0db8::0001%eth0]") == "2001:db8::1"
    assert sb.normalize_ip(" 10.0.0.1 ") == "10.0.0.1"
    assert sb.normalize_ip("host.internal") == "host.internal"

    leader = "vllm-lws-2.vllm-lws.default.svc.cluster.local"
    assert sb.lws_pod_hostname(leader, 0) == leader
    assert (
        sb.lws_pod_hostname(leader, 2)
        == "vllm-lws-2-2.vllm-lws.default.svc.cluster.local"
    )
    with pytest.raises(ValueError, match="LWS_LEADER_ADDRESS"):
        sb.lws_pod_hostname("invalid-no-dot", 1)


def test_resolve_local_slicebuilder_host_ipv4_passthrough(monkeypatch):
    def _fail_dns(*args, **kwargs):
        raise AssertionError("IPv4 should not trigger DNS lookups")

    monkeypatch.setattr(socket, "getaddrinfo", _fail_dns)
    assert sb.resolve_local_slicebuilder_host("10.0.0.1") == "10.0.0.1"


@pytest.mark.parametrize(
    "worker_idx,expected_host",
    [
        ("0", "vllm-lws-2.vllm-lws.default.svc.cluster.local"),
        ("1", "vllm-lws-2-1.vllm-lws.default.svc.cluster.local"),
    ],
)
def test_resolve_local_slicebuilder_host_lws_multinic(
    monkeypatch, worker_idx, expected_host
):
    leader_dns = "vllm-lws-2.vllm-lws.default.svc.cluster.local"
    monkeypatch.setenv("LWS_LEADER_ADDRESS", leader_dns)
    monkeypatch.setenv("LWS_WORKER_INDEX", worker_idx)
    # Multi-NIC: LWS DNS resolves to pod NIC (2001:db8:1::*), distinct from the
    # ULA data-plane local IP (fd00:1::*).
    dns_map = {
        "vllm-lws-2.vllm-lws.default.svc.cluster.local": "2001:db8:1::10",
        "vllm-lws-2-1.vllm-lws.default.svc.cluster.local": "2001:db8:1::11",
    }
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **kw: (
            [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (dns_map[host], 0, 0, 0))]
            if host in dns_map
            else (_ for _ in ()).throw(socket.gaierror(f"unknown {host}"))
        ),
    )
    assert sb.resolve_local_slicebuilder_host("fd00:1::3") == expected_host


def test_resolve_local_slicebuilder_host_fqdn_rejects_loopback_then_uses_ptr(
    monkeypatch,
):
    monkeypatch.setattr(socket, "getfqdn", lambda: "localhost")
    monkeypatch.setattr(socket, "gethostname", lambda: "node-loopback")
    monkeypatch.setattr(
        socket,
        "getnameinfo",
        lambda sockaddr, flags: ("stateful-1.svc.cluster.local", "0"),
    )

    def _fake_getaddrinfo(host, *args, **kwargs):
        if host == "node-loopback":
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0))]
        if host == "stateful-1.svc.cluster.local":
            return [
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::40", 0, 0, 0))
            ]
        raise socket.gaierror(f"unknown {host}")

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    assert (
        sb.resolve_local_slicebuilder_host("2001:db8::40")
        == "stateful-1.svc.cluster.local"
    )


def test_resolve_local_slicebuilder_host_uses_non_loopback_fqdn(monkeypatch):
    monkeypatch.setattr(socket, "getfqdn", lambda: "tpu-host-0.internal")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **kw: (
            [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::9", 0, 0, 0))]
            if host == "tpu-host-0.internal"
            else (_ for _ in ()).throw(socket.gaierror(f"unknown {host}"))
        ),
    )
    assert sb.resolve_local_slicebuilder_host("2001:db8::9") == "tpu-host-0.internal"


def test_resolve_local_slicebuilder_host_raises_when_unresolvable(monkeypatch):
    monkeypatch.setattr(socket, "getfqdn", lambda: "localhost")
    monkeypatch.setattr(socket, "gethostname", lambda: "localhost")
    monkeypatch.setattr(
        socket,
        "getnameinfo",
        lambda sockaddr, flags: (_ for _ in ()).throw(socket.gaierror("no PTR")),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **kw: (_ for _ in ()).throw(socket.gaierror("no DNS")),
    )
    with pytest.raises(RuntimeError, match="Failed to resolve local IPv6"):
        sb.resolve_local_slicebuilder_host("2001:db8::dead")


def _fake_dns(monkeypatch, dns_map):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **kw: (
            [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (dns_map[host], 0, 0, 0))]
            if host in dns_map
            else (_ for _ in ()).throw(socket.gaierror(f"unknown {host}"))
        ),
    )


def test_validate_slicebuilder_hosts_accepts_distinct_and_rejects_duplicates(
    monkeypatch,
):
    _fake_dns(
        monkeypatch,
        {
            "host-0.svc.cluster.local": "2001:db8::1",
            "host-1.svc.cluster.local": "2001:db8::2",
        },
    )
    assert sb.validate_slicebuilder_hosts(
        ["2001:db8::1", "2001:db8::2"],
        ["host-0.svc.cluster.local", "host-1.svc.cluster.local"],
    ) == ["host-0.svc.cluster.local", "host-1.svc.cluster.local"]

    with pytest.raises(ValueError, match="same slicebuilder host"):
        sb.validate_slicebuilder_hosts(
            ["2001:db8::1", "2001:db8::2"],
            ["shared-host.internal", "SHARED-HOST.internal."],
        )

    with pytest.raises(ValueError, match="loopback/unusable"):
        sb.validate_slicebuilder_hosts(
            ["2001:db8::1", "2001:db8::2"],
            ["localhost", "host-1.internal"],
        )

    with pytest.raises(ValueError, match="colon-free"):
        sb.validate_slicebuilder_hosts(
            ["2001:db8::1", "2001:db8::2"],
            ["2001:db8::1", "host-1.internal"],
        )

    with pytest.raises(ValueError, match="does not match"):
        sb.validate_slicebuilder_hosts(["10.0.0.1", "10.0.0.2"], ["10.0.0.1"])


def test_validate_slicebuilder_hosts_rejects_name_only_its_owner_resolves(
    monkeypatch,
):
    # host-1 is only in its owner's /etc/hosts: this peer cannot resolve it,
    # and a name resolving to loopback is rejected too.
    _fake_dns(
        monkeypatch,
        {"host-0.svc.cluster.local": "2001:db8::1", "host-2.local": "::1"},
    )
    monkeypatch.setattr(sb, "_PEER_RESOLVE_TIMEOUT_S", 0.0)
    with pytest.raises(ValueError, match="host-1.internal.*host-2.local"):
        sb.validate_slicebuilder_hosts(
            ["2001:db8::1", "2001:db8::2", "2001:db8::3"],
            ["host-0.svc.cluster.local", "host-1.internal", "host-2.local"],
        )


def test_validate_slicebuilder_hosts_retries_until_peer_dns_publishes(monkeypatch):
    lookups = {"host-1.svc.cluster.local": 0}
    real_dns = {
        "host-0.svc.cluster.local": "2001:db8::1",
        "host-1.svc.cluster.local": "2001:db8::2",
    }

    def _getaddrinfo(host, *a, **kw):
        if host in lookups:
            lookups[host] += 1
            if lookups[host] < 3:
                raise socket.gaierror("NXDOMAIN")
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (real_dns[host], 0, 0, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)
    sleeps: list[float] = []
    monkeypatch.setattr(sb.time, "sleep", sleeps.append)
    assert sb.validate_slicebuilder_hosts(
        ["2001:db8::1", "2001:db8::2"],
        ["host-0.svc.cluster.local", "host-1.svc.cluster.local"],
    ) == ["host-0.svc.cluster.local", "host-1.svc.cluster.local"]
    assert lookups["host-1.svc.cluster.local"] == 3
    assert sleeps == [sb._PEER_RESOLVE_RETRY_INTERVAL_S] * 2


def test_node_identity_probe_pickles_by_value_without_heavy_imports():
    probe = sb.make_node_identity_probe()
    assert probe.__module__ == "__main__"
    blob = cloudpickle.dumps(probe)

    script = (
        "import os, sys, ray.cloudpickle as cp\n"
        f"fn = cp.loads({blob!r})\n"
        "os.environ['LWS_WORKER_INDEX'] = '3'\n"
        "res = fn('10.0.0.9', '')\n"
        "assert res == (3, '10.0.0.9'), res\n"
        "heavy = [m for m in sys.modules if any(k in m for k in "
        "('vllm', 'torch', 'libtpu', 'jax'))]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
