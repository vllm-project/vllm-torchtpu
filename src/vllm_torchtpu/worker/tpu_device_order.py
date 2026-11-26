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
"""Ascending global-device-ID rank and chip binding for TPU slices."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PROBE_PREFIX = "VLLM_TPU_DEVICE_ORDER_PROBE "
_PROBE_TIMEOUT_S = 90.0

# Disposable single-chip query executed across all slice hosts before workers
# spawn so RayDistributedExecutorV2 can align global ranks 0..N-1 to physical
# global TPU device IDs 0..N-1. Bypass interpreter teardown via os._exit so
# libtpu/JAX atexit handlers cannot hang after printing the device ID.
_PROBE_SCRIPT = r"""
import json
import os
import sys
import traceback

prefix = "VLLM_TPU_DEVICE_ORDER_PROBE "
payload = {
    "rank": int(os.environ["RANK"]),
    "local_rank": int(os.environ["LOCAL_RANK"]),
}
try:
    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401
    from torch_tpu._internal.distributed import tpu_distributed

    backend = sys.argv[1]
    init_method = (
        f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    )
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )

    torch.empty((1,), device="tpu").cpu()
    payload["device_id"] = int(tpu_distributed.global_device_id())

    if dist.is_initialized():
        dist.barrier()
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)
    os._exit(0)
except BaseException as exc:
    payload["error"] = repr(exc)
    traceback.print_exc()
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)
    os._exit(1)
"""


@dataclass(frozen=True)
class AscendingDeviceOrder:
    host_order: tuple[str, ...]
    local_rank_by_host: dict[str, tuple[int, ...]]
    observed_device_ids: tuple[int, ...]


def should_use_ascending_device_order(
    *,
    mode: str,
    world_size: int,
    dp_size: int,
    pp_size: int,
    pcp_size: int,
    dcp_size: int,
    has_explicit_bundle_indices: bool,
    has_explicit_device_override: bool,
) -> bool:
    if (
        world_size <= 1
        or dp_size > 1
        or pp_size > 1
        or pcp_size > 1
        or has_explicit_bundle_indices
        or has_explicit_device_override
    ):
        return False
    normalized = mode.strip().lower()
    if normalized in ("0", "false", "off", "no"):
        return False
    if normalized in ("1", "true", "on", "yes"):
        return True
    if normalized == "auto":
        return dcp_size > 1
    raise ValueError(
        f"Invalid ascending device order mode={mode!r}; expected 'auto', '1', or '0'."
    )


def plan_ascending_device_order(
    host_devices: Mapping[str, Sequence[tuple[int, int]]],
    *,
    observed_host_order: Sequence[str] | None = None,
) -> AscendingDeviceOrder | None:
    """Resolve host order and per-host physical LOCAL_RANK for IDs 0..N-1."""
    if not host_devices:
        return None

    per_host_maps: dict[str, dict[int, int]] = {}
    all_device_ids: list[int] = []
    for ip, pairs in host_devices.items():
        if not pairs:
            return None
        local_ranks = [int(lr) for lr, _ in pairs]
        if sorted(local_ranks) != list(range(len(pairs))):
            return None
        dev_to_local: dict[int, int] = {}
        for lr, dev_id in pairs:
            dev_int = int(dev_id)
            if dev_int in dev_to_local:
                return None
            dev_to_local[dev_int] = int(lr)
        per_host_maps[ip] = dev_to_local
        all_device_ids.extend(dev_to_local.keys())

    world_size = len(all_device_ids)
    if sorted(all_device_ids) != list(range(world_size)):
        return None

    sorted_hosts = tuple(
        sorted(per_host_maps.keys(), key=lambda ip: min(per_host_maps[ip].keys()))
    )
    local_rank_by_host: dict[str, tuple[int, ...]] = {}
    offset = 0
    for ip in sorted_hosts:
        dev_to_local = per_host_maps[ip]
        count = len(dev_to_local)
        expected_ids = range(offset, offset + count)
        if sorted(dev_to_local.keys()) != list(expected_ids):
            return None
        local_rank_by_host[ip] = tuple(dev_to_local[d] for d in expected_ids)
        offset += count

    obs_hosts = (
        tuple(observed_host_order)
        if observed_host_order is not None
        else tuple(host_devices.keys())
    )
    observed: list[int] = []
    for ip in obs_hosts:
        by_local = {int(lr): int(dev_id) for lr, dev_id in host_devices[ip]}
        observed.extend(by_local[lr] for lr in range(len(by_local)))

    return AscendingDeviceOrder(
        host_order=sorted_hosts,
        local_rank_by_host=local_rank_by_host,
        observed_device_ids=tuple(observed),
    )


def reorder_bundles_by_host_order(
    bundle_to_node_id: Sequence[tuple[int, str, str]],
    host_order: Sequence[str],
) -> list[tuple[int, str, str]]:
    """Reorder placement-group bundles by host IP while preserving local order."""
    by_ip: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for entry in bundle_to_node_id:
        by_ip[entry[2]].append(entry)
    unknown = [ip for ip in by_ip if ip not in set(host_order)]
    if unknown:
        raise ValueError(
            f"Placement-group hosts {unknown} missing from target host_order "
            f"{list(host_order)}."
        )
    reordered: list[tuple[int, str, str]] = []
    for ip in host_order:
        reordered.extend(by_ip.get(ip, ()))
    if len(reordered) != len(bundle_to_node_id):
        raise ValueError(
            f"Reordered bundle count {len(reordered)} != input count "
            f"{len(bundle_to_node_id)}."
        )
    return reordered


def _parse_query_output(log_text: str) -> dict[str, object] | None:
    for line in log_text.splitlines():
        if line.startswith(_PROBE_PREFIX):
            return json.loads(line[len(_PROBE_PREFIX) :])
    return None


def probe_host_tpu_device_ids(
    *,
    rank_offset: int,
    local_world_size: int,
    world_size: int,
    num_nodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    topology: str,
    slicebuilder_addresses: str,
    backend: str,
    xprof_session_id: str,
    timeout_s: float = _PROBE_TIMEOUT_S,
    probe_script: str = _PROBE_SCRIPT,
) -> list[tuple[int, int]]:
    """Query (local_rank, global_device_id) for every chip on this host."""
    base_env = os.environ.copy()
    for key in (
        "TPU_VISIBLE_CHIPS",
        "TPU_VISIBLE_DEVICES",
    ):
        base_env.pop(key, None)
    base_env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"

    with (
        tempfile.TemporaryDirectory(prefix="vllm-tpu-devorder-check-") as tmpdir,
        contextlib.ExitStack() as stack,
    ):
        procs: list[tuple[int, subprocess.Popen[bytes], str]] = []
        try:
            for local_rank in range(local_world_size):
                rank = rank_offset + local_rank
                env = base_env.copy()
                env.update(
                    {
                        "RANK": str(rank),
                        "LOCAL_RANK": str(local_rank),
                        "WORLD_SIZE": str(world_size),
                        "LOCAL_WORLD_SIZE": str(local_world_size),
                        "NNODES": str(num_nodes),
                        "NODE_RANK": str(node_rank),
                        "TPU_NUM_HOSTS": str(num_nodes),
                        "MASTER_ADDR": master_addr,
                        "MASTER_PORT": str(master_port),
                        "TORCH_TPU_TOPOLOGY": topology,
                        "TORCH_TPU_SLICEBUILDER_ADDRESSES": slicebuilder_addresses,
                        "TORCH_TPU_XPROF_SESSION_ID": xprof_session_id,
                    }
                )
                log_path = os.path.join(tmpdir, f"rank-{rank}.log")
                log_file = stack.enter_context(open(log_path, "wb"))
                proc = subprocess.Popen(
                    [sys.executable, "-c", probe_script, backend],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                procs.append((local_rank, proc, log_path))

            deadline = time.monotonic() + timeout_s
            timed_out = False
            while True:
                if all(proc.poll() is not None for _, proc, _ in procs):
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    for _, proc, _ in procs:
                        if proc.poll() is None:
                            proc.kill()
                    for _, proc, _ in procs:
                        proc.wait(timeout=5.0)
                    break
                time.sleep(0.1)
        finally:
            for _, proc, _ in procs:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5.0)

        results: list[tuple[int, int]] = []
        errors: list[str] = []
        for local_rank, proc, log_path in procs:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            payload = _parse_query_output(text)
            if proc.returncode != 0 or payload is None or "device_id" not in payload:
                tail = text[-600:] if text else "<empty>"
                errors.append(f"local_rank={local_rank} rc={proc.returncode}: {tail}")
                continue
            lr_raw: Any = payload["local_rank"]
            did_raw: Any = payload["device_id"]
            results.append((int(lr_raw), int(did_raw)))

        if timed_out or errors:
            raise RuntimeError(
                f"TPU device-order discovery failed on node_rank={node_rank} "
                f"(timed_out={timed_out}): {'; '.join(errors[:2])}"
            )
        return results
