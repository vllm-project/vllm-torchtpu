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
"""Colon-free slicebuilder host resolution and validation for IPv6 TPU slices.

TODO(https://github.com/google-pytorch/torch_tpu/issues/7040): remove this
hostname substitution workaround once TorchTPU accepts bracketed
``[ipv6]:port`` literals in ``TORCH_TPU_SLICEBUILDER_ADDRESSES``.
"""

import ipaddress
import socket
import time
from collections.abc import Callable

from vllm_torchtpu import envs

# A pod's own name is in its /etc/hosts before cluster DNS publishes it, and an
# early miss can be negatively cached, so peers retry failed lookups briefly.
_PEER_RESOLVE_TIMEOUT_S = 30.0
_PEER_RESOLVE_RETRY_INTERVAL_S = 2.0

_LOOPBACK_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }
)


def normalize_ip(addr: str) -> str:
    """Return canonical string form for IPv4/IPv6 literals, stripping scope IDs."""
    raw = addr.strip().strip("[]").split("%", 1)[0]
    try:
        return str(ipaddress.ip_address(raw))
    except ValueError:
        return addr.strip()


def lws_pod_hostname(leader_address: str, worker_index: int) -> str:
    """Derive the DNS name for ``worker_index`` in an LWS replica group."""
    leader_pod, sep, service_domain = leader_address.partition(".")
    if not leader_pod or not sep or not service_domain:
        raise ValueError(
            "Invalid LWS_LEADER_ADDRESS: expected '<leader-pod>.<service-domain>', "
            f"got {leader_address!r}."
        )
    if worker_index == 0:
        return leader_address
    return f"{leader_pod}-{worker_index}.{service_domain}"


def _is_usable_hostname(hostname: str) -> bool:
    name = hostname.strip().rstrip(".").lower()
    return bool(name) and ":" not in name and name not in _LOOPBACK_HOSTNAMES


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _resolves_to_non_loopback(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for info in infos:
        addr = info[4][0] if info[4] else None
        if not isinstance(addr, str):
            continue
        try:
            if not ipaddress.ip_address(normalize_ip(addr)).is_loopback:
                return True
        except ValueError:
            continue
    return False


def _unresolvable_hostnames(hosts: list[str]) -> list[str]:
    """Return the hostnames this process cannot resolve to a non-loopback address."""
    pending = [h for h in hosts if not _is_ip_literal(h)]
    deadline = time.monotonic() + _PEER_RESOLVE_TIMEOUT_S
    while True:
        pending = [h for h in pending if not _resolves_to_non_loopback(h)]
        if not pending or time.monotonic() >= deadline:
            return pending
        time.sleep(_PEER_RESOLVE_RETRY_INTERVAL_S)


def make_node_identity_probe() -> Callable[[str, str], tuple[int | None, str | None]]:
    """Return a stdlib-only probe closure pickled by value for remote Ray nodes.

    Returns ``(lws_worker_index, local_sb_host)`` without importing ``vllm``,
    ``vllm_torchtpu``, ``torch``, or ``libtpu`` on remote probe workers.
    """

    def _probe(ip: str, leader_address: str) -> tuple[int | None, str | None]:
        import ipaddress as _ipaddress
        import os as _os
        import socket as _socket

        loopback_names = (
            "localhost",
            "localhost.localdomain",
            "ip6-localhost",
            "ip6-loopback",
        )
        raw_idx = _os.getenv("LWS_WORKER_INDEX")
        worker_idx: int | None = None
        if raw_idx is not None and raw_idx.strip() != "":
            try:
                parsed = int(raw_idx)
                if parsed >= 0:
                    worker_idx = parsed
            except ValueError:
                worker_idx = None

        if ":" not in ip:
            return worker_idx, ip

        def _norm(addr: str) -> str:
            raw = addr.strip().strip("[]").split("%", 1)[0]
            try:
                return str(_ipaddress.ip_address(raw))
            except ValueError:
                return addr.strip()

        def _resolves(hostname: str, target_ip: str | None = None) -> bool:
            name = hostname.strip().rstrip(".").lower()
            if not name or ":" in name or name in loopback_names:
                return False
            try:
                infos = _socket.getaddrinfo(hostname, None, type=_socket.SOCK_STREAM)
            except OSError:
                return False
            for info in infos:
                sockaddr = info[4]
                if not sockaddr or not isinstance(sockaddr[0], str):
                    continue
                addr_norm = _norm(sockaddr[0])
                try:
                    if _ipaddress.ip_address(addr_norm).is_loopback:
                        continue
                except ValueError:
                    continue
                if target_ip is None or addr_norm == target_ip:
                    return True
            return False

        norm_ip = _norm(ip)
        leader = leader_address.strip()
        if leader and worker_idx is not None:
            leader_pod, sep, domain = leader.partition(".")
            if leader_pod and sep and domain:
                candidate = (
                    leader if worker_idx == 0 else f"{leader_pod}-{worker_idx}.{domain}"
                )
                if _resolves(candidate):
                    return worker_idx, candidate

        for candidate in (_socket.getfqdn(), _socket.gethostname()):
            if _resolves(candidate):
                return worker_idx, candidate

        try:
            ptr_host = _socket.getnameinfo((norm_ip, 0), _socket.NI_NAMEREQD)[0]
        except OSError:
            ptr_host = ""
        if ptr_host and _resolves(ptr_host, norm_ip):
            return worker_idx, ptr_host

        return worker_idx, None

    # Detach from vllm_torchtpu so cloudpickle never references this package.
    _probe.__module__ = "__main__"
    return _probe


def resolve_local_slicebuilder_host(local_ip: str) -> str:
    """Return a colon-free slicebuilder host published by this host.

    IPv4 literals pass through unchanged. IPv6 literals are replaced by a
    forward-resolvable, non-loopback hostname known to this host (LWS pod DNS
    from ``LWS_LEADER_ADDRESS`` + ``LWS_WORKER_INDEX``, local FQDN/hostname, or
    reverse DNS PTR).
    """
    if ":" not in local_ip:
        return local_ip

    leader_address = envs.LWS_LEADER_ADDRESS
    worker_index = envs.LWS_WORKER_INDEX
    if leader_address and worker_index is not None:
        # Validate LWS_LEADER_ADDRESS format upfront for clear diagnostics.
        lws_pod_hostname(leader_address, worker_index)

    _, resolved = make_node_identity_probe()(local_ip, leader_address)
    if resolved is not None:
        return resolved

    raise RuntimeError(
        f"Failed to resolve local IPv6 slicebuilder host for {local_ip!r} to a "
        "forward-resolvable, non-loopback DNS hostname via LWS_LEADER_ADDRESS "
        f"({leader_address!r}), local hostname/FQDN, or reverse DNS."
    )


def validate_slicebuilder_hosts(host_ips: list[str], sb_hosts: list[str]) -> list[str]:
    """Validate per-host slicebuilder names and reject duplicates or IPv6 literals."""
    if len(host_ips) != len(sb_hosts):
        raise ValueError(
            f"Slicebuilder host count ({len(sb_hosts)}) does not match host IP "
            f"count ({len(host_ips)})."
        )

    cleaned: list[str] = []
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for idx, (ip, raw_host) in enumerate(zip(host_ips, sb_hosts)):
        host = raw_host.strip()
        if not host or ":" in host:
            raise ValueError(
                f"Host {idx} ({ip!r}) published invalid slicebuilder host "
                f"{raw_host!r}: expected a non-empty colon-free hostname or IPv4."
            )
        if ":" in ip and not _is_usable_hostname(host):
            raise ValueError(
                f"Host {idx} ({ip!r}) published loopback/unusable slicebuilder "
                f"host {host!r}."
            )
        key = host.rstrip(".").lower()
        if key in seen:
            duplicates.append(host)
        else:
            seen[key] = idx
        cleaned.append(host)

    if duplicates:
        raise ValueError(
            "Multiple slice hosts resolved to the same slicebuilder host "
            f"{sorted(set(duplicates))} (full list: {cleaned}); each host "
            "must publish a distinct address."
        )

    # Each host checked only that its own resolver knows its name; peers must
    # resolve it too (an /etc/hosts-only name would pass the owner's check).
    unresolvable = _unresolvable_hostnames(cleaned)
    if unresolvable:
        raise ValueError(
            f"Slicebuilder hosts {unresolvable} do not resolve to a non-loopback "
            f"address from this host after {_PEER_RESOLVE_TIMEOUT_S:.0f}s; every "
            "slice host must be resolvable through DNS shared by all peers."
        )
    return cleaned
