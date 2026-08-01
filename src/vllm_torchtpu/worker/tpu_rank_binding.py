# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Mapping

import portpicker

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_PCP_LOCAL_RANK_REMAPS: dict[int, tuple[tuple[int, ...], str]] = {}
_PCP_REMAP_PROBE_PREFIX = "VLLM_TPU_PCP_REMAP_PROBE "
_PCP_REMAP_PROBE_TIMEOUT_S = float(
    os.getenv("TPU_PCP_REMAP_PROBE_TIMEOUT_S", "120.0"))
_PCP_REMAP_PROBE_SCRIPT = r"""
import json
import os
import sys
import traceback

prefix = "VLLM_TPU_PCP_REMAP_PROBE "
payload = {
    "rank": int(os.environ["RANK"]),
    "local_rank": int(os.environ["LOCAL_RANK"]),
}
try:
    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401
    from torch_tpu._internal.distributed import tpu_distributed

    init_method = (
        f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="tpu_dist",
            init_method=init_method,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )

    # Touch TPU before querying TorchTPU/JAX binding metadata.
    torch.empty((1,), device="tpu").cpu()
    import jax
    payload["global_device_id"] = int(tpu_distributed.global_device_id())
    payload["jax_device_ids"] = [int(device.id) for device in jax.devices()]

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)
except BaseException as exc:
    payload["error"] = repr(exc)
    traceback.print_exc()
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)
    sys.exit(1)
"""


@dataclass(frozen=True)
class TpuWorkerBinding:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    init_rank: int
    init_world_size: int
    init_local_rank: int
    native_local_rank: int
    dp_rank: int = 0
    dp_size: int = 1
    local_rank_offset: int = 0
    pcp_local_rank_remap: tuple[int, ...] | None = None
    pcp_remap_source: str | None = None

    def as_env(self) -> dict[str, str]:
        return {
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.local_rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }


def _get_int_env(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, "") or default)
    except ValueError:
        return default


def _validate_pcp_local_rank_remap(world_size: int,
                                   remap: Sequence[int]) -> tuple[int, ...]:
    remap_tuple = tuple(int(item) for item in remap)
    if (len(remap_tuple) != world_size
            or sorted(remap_tuple) != list(range(world_size))):
        raise ValueError("PCP local rank remap must be a permutation of "
                         f"0..{world_size - 1}, got {remap_tuple}.")
    return remap_tuple


def compute_pcp_local_rank_remap(
        local_rank_to_device_id: Sequence[int],
        jax_device_ids: Sequence[int]) -> tuple[int, ...]:
    """Return native PCP rank -> TorchTPU LOCAL_RANK remap."""
    local_ids = tuple(int(device_id) for device_id in local_rank_to_device_id)
    jax_ids = tuple(int(device_id) for device_id in jax_device_ids)
    if len(local_ids) != len(jax_ids):
        raise RuntimeError(
            "PCP remap probe produced mismatched result lengths: "
            f"local_rank_to_device_id={local_ids}, jax_device_ids={jax_ids}.")
    if len(set(local_ids)) != len(local_ids):
        raise RuntimeError(
            "PCP remap probe produced duplicate TorchTPU device ids: "
            f"{local_ids}.")
    if len(set(jax_ids)) != len(jax_ids):
        raise RuntimeError(
            f"PCP remap probe produced duplicate JAX device ids: {jax_ids}.")

    device_to_local_rank = {
        device_id: local_rank
        for local_rank, device_id in enumerate(local_ids)
    }
    missing = tuple(device_id for device_id in jax_ids
                    if device_id not in device_to_local_rank)
    if missing:
        raise RuntimeError(
            "PCP remap probe JAX device ids are missing from TorchTPU "
            f"bindings: missing={missing}, local_rank_to_device_id="
            f"{local_ids}, jax_device_ids={jax_ids}.")
    return tuple(device_to_local_rank[device_id] for device_id in jax_ids)


def set_pcp_local_rank_remap(world_size: int, remap: Sequence[int], *,
                             source: str) -> None:
    _PCP_LOCAL_RANK_REMAPS[int(world_size)] = (
        _validate_pcp_local_rank_remap(int(world_size), remap),
        source,
    )


def clear_pcp_local_rank_remaps() -> None:
    _PCP_LOCAL_RANK_REMAPS.clear()


def has_pcp_local_rank_remap(world_size: int) -> bool:
    return int(world_size) in _PCP_LOCAL_RANK_REMAPS


def get_pcp_native_rank_local_rank_remap(
        world_size: int) -> tuple[tuple[int, ...], str]:
    """Return native PCP rank -> TorchTPU LOCAL_RANK mapping."""
    try:
        return _PCP_LOCAL_RANK_REMAPS[int(world_size)]
    except KeyError as exc:
        raise RuntimeError("PCP LOCAL_RANK remap has not been probed for "
                           f"world_size={world_size}.") from exc


def _get_default_tpu_topology(world_size: int) -> str:
    from torch_tpu._internal.utils import hardware

    topology = hardware.get_tpu_topology(world_size)
    if topology is None:
        raise ValueError("No TPU devices found.")
    return topology


def probe_pcp_local_rank_remap(
        world_size: int, *, get_topology: Callable[[int],
                                                   str]) -> tuple[int, ...]:
    master_port = portpicker.pick_unused_port()
    sb_ports = [portpicker.pick_unused_port() for _ in range(world_size)]
    sb_addresses = ",".join(f"localhost:{port}" for port in sb_ports)
    topology = os.environ.get("TORCH_TPU_TOPOLOGY") or get_topology(world_size)

    logger.info("Probing PCP LOCAL_RANK remap for world_size=%d topology=%s.",
                world_size, topology)
    with tempfile.TemporaryDirectory(
            prefix="vllm-tpu-pcp-remap-probe-") as tmpdir:
        procs: list[tuple[int, subprocess.Popen, object, str]] = []
        for rank in range(world_size):
            env = os.environ.copy()
            env.update({
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_WORLD_SIZE": str(world_size),
                "GROUP_RANK": "0",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(master_port),
                "TORCH_TPU_TOPOLOGY": topology,
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb_addresses,
                "TORCH_TPU_XPROF_SESSION_ID": str(time.time_ns()),
            })
            log_path = os.path.join(tmpdir, f"rank-{rank}.log")
            # errors="replace": verbose libtpu output (e.g. under TPU_VMODULE)
            # can contain non-UTF8 bytes; a strict read kills the server
            # inside VllmConfig validation.
            log_file = open(log_path, "w+", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                [sys.executable, "-c", _PCP_REMAP_PROBE_SCRIPT],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            procs.append((rank, proc, log_file, log_path))

        deadline = time.monotonic() + _PCP_REMAP_PROBE_TIMEOUT_S
        timed_out = False
        while True:
            if all(proc.poll() is not None for _, proc, _, _ in procs):
                break
            if time.monotonic() >= deadline:
                timed_out = True
                for _, proc, _, _ in procs:
                    if proc.poll() is None:
                        proc.kill()
                break
            time.sleep(0.1)

        payloads: dict[int, dict] = {}
        logs: list[str] = []
        failed_returncodes: dict[int, int] = {}
        for rank, proc, log_file, log_path in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                timed_out = True
                if proc.poll() is None:
                    proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            log_file.seek(0)
            log_text = log_file.read()
            log_file.close()
            logs.append(f"rank {rank} log ({log_path}):\n{log_text}")
            for line in log_text.splitlines():
                if line.startswith(_PCP_REMAP_PROBE_PREFIX):
                    payload = json.loads(line[len(_PCP_REMAP_PROBE_PREFIX):])
                    payloads[int(payload["local_rank"])] = payload
            if proc.returncode is None:
                timed_out = True
            elif proc.returncode != 0:
                failed_returncodes[rank] = int(proc.returncode)
        if timed_out or any(code == -9
                            for code in failed_returncodes.values()):
            raise RuntimeError("PCP LOCAL_RANK remap probe timed out.\n" +
                               "\n".join(logs))
        if failed_returncodes:
            raise RuntimeError(
                "PCP LOCAL_RANK remap probe failed with returncodes "
                f"{failed_returncodes}.\n" + "\n".join(logs))
        expected_ranks = set(range(world_size))
        if set(payloads) != expected_ranks:
            raise RuntimeError(
                "PCP LOCAL_RANK remap probe did not report every rank: "
                f"expected={sorted(expected_ranks)}, got={sorted(payloads)}.")
        errored = [
            payload for payload in payloads.values() if "error" in payload
        ]
        if errored:
            raise RuntimeError("PCP LOCAL_RANK remap probe reported errors: "
                               f"{errored}.")

        ordered_payloads = [
            payloads[local_rank] for local_rank in range(world_size)
        ]
        local_rank_to_device_id = tuple(
            int(payload["global_device_id"]) for payload in ordered_payloads)
        first_jax_device_ids = tuple(
            int(device_id)
            for device_id in ordered_payloads[0]["jax_device_ids"])
        for payload in ordered_payloads[1:]:
            jax_device_ids = tuple(
                int(device_id) for device_id in payload["jax_device_ids"])
            if jax_device_ids != first_jax_device_ids:
                raise RuntimeError(
                    "PCP LOCAL_RANK remap probe saw inconsistent JAX device "
                    "orders across ranks: "
                    f"rank0={first_jax_device_ids}, "
                    f"rank{payload['local_rank']}={jax_device_ids}.")
        return compute_pcp_local_rank_remap(local_rank_to_device_id,
                                            first_jax_device_ids[:world_size])


def ensure_pcp_local_rank_remap(world_size: int, *,
                                get_topology: Callable[[int], str]) -> None:
    if has_pcp_local_rank_remap(world_size):
        return

    remap = probe_pcp_local_rank_remap(world_size, get_topology=get_topology)
    set_pcp_local_rank_remap(
        world_size,
        remap,
        source=f"dynamic probe world_size={world_size}",
    )
    logger.info("Dynamic PCP LOCAL_RANK remap for world_size=%d: %s.",
                world_size, remap)


def get_pcp_worker_local_rank_env(
    native_local_rank: int,
    local_world: int,
    pcp_size: int,
) -> tuple[int, tuple[int, ...] | None, str]:
    if pcp_size <= 1:
        return native_local_rank, None, "PCP disabled"

    if not has_pcp_local_rank_remap(local_world):
        ensure_pcp_local_rank_remap(local_world,
                                    get_topology=_get_default_tpu_topology)
    remap, remap_source = get_pcp_native_rank_local_rank_remap(local_world)
    if not 0 <= native_local_rank < len(remap):
        raise ValueError(
            f"native_local_rank={native_local_rank} is out of range for "
            f"PCP local rank remap {remap}.")
    return remap[native_local_rank], remap, remap_source


def _get_spawned_pcp_local_rank(
    *,
    env: Mapping[str, str],
    local_rank_offset: int,
    local_world: int,
) -> tuple[int, None, str]:
    local_rank_env = _get_int_env(env, "LOCAL_RANK", local_rank_offset)
    local_rank = local_rank_env - local_rank_offset
    if not 0 <= local_rank < local_world:
        raise ValueError(
            f"LOCAL_RANK={local_rank_env} is out of range for PCP local "
            f"world {local_world} with offset {local_rank_offset}.")
    return local_rank, None, "spawn LOCAL_RANK env"


def get_tpu_worker_binding(
    parallel_config,
    rank: int,
    local_rank: int,
    *,
    env: Mapping[str, str],
    use_spawned_pcp_local_rank: bool = False,
) -> TpuWorkerBinding:
    world_size = parallel_config.world_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_size = (_get_int_env(env, "TORCH_TPU_DP_SIZE", 0)
               or parallel_config.data_parallel_size)
    local_rank_offset = _get_int_env(env, "TPU_LOCAL_RANK_OFFSET", 0)

    rank = int(rank)
    local_rank = int(local_rank)
    if dp_size > 1:
        dp_rank = parallel_config.data_parallel_index
        assert dp_rank is not None, (
            "ParallelConfig.data_parallel_index must be resolved when "
            "data_parallel_size > 1")

        native_local_rank = dp_rank * world_size + local_rank
        global_rank = dp_rank * world_size + rank
        global_world = world_size * dp_size
        if pcp_size > 1 and use_spawned_pcp_local_rank:
            tpu_local_rank, remap, source = _get_spawned_pcp_local_rank(
                env=env,
                local_rank_offset=local_rank_offset,
                local_world=global_world,
            )
        else:
            tpu_local_rank, remap, source = get_pcp_worker_local_rank_env(
                native_local_rank, global_world, pcp_size)
        return TpuWorkerBinding(
            rank=global_rank,
            local_rank=local_rank_offset + tpu_local_rank,
            world_size=global_world,
            local_world_size=local_rank_offset + global_world,
            init_rank=rank,
            init_world_size=world_size,
            init_local_rank=tpu_local_rank,
            native_local_rank=native_local_rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            local_rank_offset=local_rank_offset,
            pcp_local_rank_remap=remap,
            pcp_remap_source=source,
        )

    native_local_rank = local_rank
    if pcp_size > 1 and use_spawned_pcp_local_rank:
        tpu_local_rank, remap, source = _get_spawned_pcp_local_rank(
            env=env,
            local_rank_offset=local_rank_offset,
            local_world=world_size,
        )
    else:
        tpu_local_rank, remap, source = get_pcp_worker_local_rank_env(
            native_local_rank, world_size, pcp_size)
    return TpuWorkerBinding(
        rank=rank,
        local_rank=local_rank_offset + tpu_local_rank,
        world_size=world_size,
        local_world_size=local_rank_offset + world_size,
        init_rank=rank,
        init_world_size=world_size,
        init_local_rank=tpu_local_rank,
        native_local_rank=native_local_rank,
        dp_size=1,
        local_rank_offset=local_rank_offset,
        pcp_local_rank_remap=remap,
        pcp_remap_source=source,
    )
