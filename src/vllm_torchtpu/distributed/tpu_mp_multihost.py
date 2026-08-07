# SPDX-License-Identifier: Apache-2.0
"""TPU multi-host bootstrap for the vLLM `mp` (multiprocessing) executor.

When `--nnodes N` (N>1) is passed without `--distributed-executor-backend
ray`, each host runs its own `vllm serve` process (headless on all but node
0) that spawns TP-local workers via `TpuMultiprocExecutor`. TorchTPU still
needs a handful of env vars describing the *whole* multi-host ICI mesh
before those local workers are forked:

  * TORCH_TPU_SLICEBUILDER_ADDRESSES - "ip:port" for every worker on every
    host, ordered by (node_rank, local_rank), so all NNODES*LOCAL_WORLD_SIZE
    torch_tpu processes can find each other. Mirrors the address list
    ray_distributed_executor.py builds for the Ray backend.
  * TORCH_TPU_TOPOLOGY - ICI mesh shape for the full chip count, looked up
    the same way the Ray backend does (TPU_MULTIHOST_TOPOLOGY_MAP).
  * TPU_NUM_HOSTS / NODE_RANK - consumed by the multi-host KV-transfer paths.

There is no Ray actor system here to query other hosts' IPs, so a
short-lived TCPStore rendezvous (host 0 is the store master, reusing
--master-addr from the CLI) exchanges each host's IP address. TORCH_TPU_XPROF
_SESSION_ID must also match across hosts for the profiler trace merge, so it
rides along in the same rendezvous.
"""

import os
import time
from datetime import timedelta

import torch.distributed as dist
from vllm.utils.network_utils import get_ip

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_DONE_SENTINEL = "_TORCH_TPU_MP_MULTIHOST_ENV_PREPARED"

# The master's TCPStore server thread stops as soon as the TCPStore object is
# garbage-collected. Other hosts may still be reading keys from it after this
# process's rendezvous call returns (there's no signal for "everyone is
# done"), so keep the master's store alive for the process lifetime instead
# of letting it go out of scope.
_keepalive_stores: list = []


def _rendezvous_port(master_port: int) -> int:
    # A dedicated port next to MASTER_PORT so this doesn't collide with
    # vLLM's own distributed_init_method rendezvous on the same address.
    override = os.environ.get("TORCH_TPU_MP_RENDEZVOUS_PORT")
    if override:
        return int(override)
    return int(master_port) + 1


def prepare_mp_multihost_env(parallel_config) -> None:
    """Set TORCH_TPU_* env vars for a genuine multi-host `mp` backend run.

    No-op unless `--nnodes` > 1. Idempotent per-process so it is safe to call
    more than once (e.g. if check_and_update_config runs again for a
    per-engine config copy).
    """
    nnodes = parallel_config.nnodes
    if nnodes <= 1:
        return

    if os.environ.get(_DONE_SENTINEL) == str(nnodes):
        return

    node_rank = parallel_config.node_rank
    local_world_size = parallel_config.local_world_size
    total_chips = local_world_size * nnodes
    master_addr = parallel_config.master_addr
    rendezvous_port = _rendezvous_port(parallel_config.master_port)

    logger.info(
        "TPU mp multihost: rendezvousing node_rank=%d/%d "
        "local_world_size=%d master_addr=%s rendezvous_port=%d", node_rank,
        nnodes, local_world_size, master_addr, rendezvous_port)

    is_master = node_rank == 0
    store = dist.TCPStore(
        host_name=master_addr,
        port=rendezvous_port,
        world_size=nnodes,
        is_master=is_master,
        timeout=timedelta(seconds=300),
    )
    if is_master:
        # Keep the server thread alive; see module-level comment.
        _keepalive_stores.append(store)

    store.set(f"tpu_mp_host_ip_{node_rank}", get_ip())
    if is_master:
        store.set("tpu_mp_xprof_session_id", str(time.time_ns()))

    ip_keys = [f"tpu_mp_host_ip_{i}" for i in range(nnodes)]
    wait_keys = ip_keys + ["tpu_mp_xprof_session_id"]
    store.wait(wait_keys, timedelta(seconds=300))
    host_ips = [store.get(key).decode() for key in ip_keys]
    xprof_session_id = store.get("tpu_mp_xprof_session_id").decode()

    # Same construction as ray_distributed_executor.py: one address per chip,
    # ordered by (host, local_rank), spanning every host in the slice.
    base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", 8070))
    sb_addresses = [
        f"{ip}:{base_port + local_rank}" for ip in host_ips
        for local_rank in range(local_world_size)
    ]
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(sb_addresses)

    # Lazy import: tpu_platform.py imports this module from inside
    # check_and_update_config to avoid a top-level circular import.
    from vllm_torchtpu.platforms.tpu_platform import TPU_MULTIHOST_TOPOLOGY_MAP
    topology = TPU_MULTIHOST_TOPOLOGY_MAP.get(total_chips)
    if topology is None:
        raise ValueError(
            f"Cannot find topology for {total_chips} chips. The supported "
            f"number of chips are {list(TPU_MULTIHOST_TOPOLOGY_MAP.keys())}")
    os.environ["TORCH_TPU_TOPOLOGY"] = topology
    os.environ["TPU_NUM_HOSTS"] = str(nnodes)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id
    os.environ[_DONE_SENTINEL] = str(nnodes)

    logger.info(
        "TPU mp multihost: node_rank=%d host_ips=%s "
        "TORCH_TPU_SLICEBUILDER_ADDRESSES=%s TORCH_TPU_TOPOLOGY=%s", node_rank,
        host_ips, os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"], topology)
