#!/bin/bash
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
#
# Helpers shared by the runners that start a global DRAM pool control plane
# (global_registry_server and kv_cache_host_store_node_main) next to a
# vllm serve. Source this file; it runs nothing.
#
# The runner defines fail() and sets:
#   BIND_HOST      address every server binds and is probed on
#   GPC_LOG_DIR    directory holding registry.log and store_node.log
#   REGISTRY_PID, NODE_PID, SERVER_PID   once each process is started (an
#                  empty value means not started yet)
#   REGISTRY_PORT  for the bind-failure message

# Ports at or above the kernel's ephemeral range can be taken at any moment
# by an outgoing connection, so pick_port never scans into it.
EPHEMERAL_FLOOR="$(cut -f1 /proc/sys/net/ipv4/ip_local_port_range \
  2>/dev/null || echo 32768)"

# First free port at or above $1, probed the way the servers bind it: an
# IPv6 wildcard socket, which on Linux covers the IPv4 wildcard too. The
# probe closes the socket before the server opens it, so this narrows the
# race to milliseconds instead of leaving the port unchecked for the run.
# Scans at most 100 ports, so bases 500 apart never overlap.
pick_port() {
  python3 - "$1" "${EPHEMERAL_FLOOR}" <<'PY'
import socket
import sys

base, floor = int(sys.argv[1]), int(sys.argv[2])
end = min(base + 100, floor)
# gRPC binds a dualstack wildcard where it can and falls back to IPv4
# where it cannot; probe with whichever this host actually supports.
dualstack = socket.has_dualstack_ipv6()
family = socket.AF_INET6 if dualstack else socket.AF_INET
for port in range(base, end):
    try:
        probe = socket.create_server(("", port), family=family,
                                     dualstack_ipv6=dualstack)
    except OSError:
        continue
    probe.close()
    print(port)
    break
else:
    raise SystemExit(f"no port free in [{base}, {end})")
PY
}

# TERM, then after $1 seconds (default 2) KILL, the server, the store node
# and the registry. Meant for the runner's EXIT trap.
stop_gpc_processes() {
  local grace="${1:-2}" pid
  for pid in "${SERVER_PID}" "${NODE_PID}" "${REGISTRY_PID}"; do
    if [[ -n "${pid}" ]]; then kill -TERM "${pid}" 2>/dev/null || true; fi
  done
  sleep "${grace}"
  for pid in "${SERVER_PID}" "${NODE_PID}" "${REGISTRY_PID}"; do
    if [[ -n "${pid}" ]]; then kill -KILL "${pid}" 2>/dev/null || true; fi
  done
}

check_control_plane() {
  kill -0 "${REGISTRY_PID}" 2>/dev/null || {
    tail -50 "${GPC_LOG_DIR}/registry.log" || true
    # A lost port race is the one failure the registry cannot report itself
    # on older wheels: it announces the address it never bound and then
    # crashes, so name the cause here rather than leaving "exited".
    if grep -qE "Failed to (add port|listen)" \
        "${GPC_LOG_DIR}/registry.log" 2>/dev/null; then
      fail "global_registry_server could not bind port ${REGISTRY_PORT}:\
 something else on this agent holds it"
    fi
    fail "global_registry_server exited"
  }
  if [[ -n "${NODE_PID}" ]]; then
    kill -0 "${NODE_PID}" 2>/dev/null || {
      tail -50 "${GPC_LOG_DIR}/store_node.log" || true
      fail "kv_cache_host_store_node_main exited"
    }
  fi
}

# Wait up to $3 seconds (default 60) for ${BIND_HOST}:$1 to accept a
# connection; $2 names it in messages. Fails early if the control plane dies.
wait_port() {
  local port="$1" what="$2" timeout="${3:-60}" waited=0
  until (exec 3<>"/dev/tcp/${BIND_HOST}/${port}") 2>/dev/null; do
    exec 3>&- 2>/dev/null || true
    check_control_plane
    [[ "${waited}" -lt "${timeout}" ]] \
      || fail "${what} (${BIND_HOST}:${port}) not reachable after ${waited}s"
    sleep 1
    waited=$((waited + 1))
  done
  exec 3>&- 2>/dev/null || true
  echo "${what} port ${port} reachable"
}
