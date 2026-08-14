# SPDX-License-Identifier: Apache-2.0
"""TPU multi-host bootstrap for the vLLM `mp` (multiprocessing) executor.

vLLM has two independent mechanisms for spreading a deployment across
multiple hosts without Ray, and this module has a bootstrap function for
each:

  * `--nnodes N` (N>1): TP/PP/PCP sharded across hosts. Each host runs its
    own `vllm serve` process (headless on all but node 0) that spawns its
    local shard's workers via `TpuMultiprocExecutor`.
    See `prepare_mp_multihost_env`.
  * `--data-parallel-size` wider than `--data-parallel-size-local`: DP
    replicas sharded across hosts (vLLM's native hybrid-LB launch, secondary
    hosts pass `--data-parallel-start-rank`). `--nnodes` is not involved;
    each host again runs its own `vllm serve` process, but discovers its
    slice of DP ranks from data_parallel_size_local/data_parallel_start_rank.
    See `prepare_mp_multihost_dp_env`.

In both cases TorchTPU needs a handful of env vars describing the *whole*
multi-host ICI mesh before local workers are forked:

  * TORCH_TPU_SLICEBUILDER_ADDRESSES - "ip:port" for every worker on every
    host, ordered by increasing global rank, so all torch_tpu processes in
    the mesh can find each other. Mirrors the address list
    ray_distributed_executor.py builds for the Ray backend.
  * TORCH_TPU_TOPOLOGY - ICI mesh shape for the full chip count, looked up
    the same way the Ray backend does (get_tpu_multihost_topology).

There is no Ray actor system here to query other hosts' IPs, so a
short-lived TCPStore rendezvous (host 0 is the store master) exchanges each
host's IP address. TORCH_TPU_XPROF_SESSION_ID must also match across hosts
for the profiler trace merge, so it rides along in the same rendezvous.

Both bootstrap functions are called from tpu_platform.py's
check_and_update_config rather than from TpuMultiprocExecutor._init_executor
(where the Ray backend does the equivalent setup) because the two backends
launch the headless node differently. Ray has no separate headless worker
process: one driver creates the placement group and actors across the whole
cluster from inside its executor, so "runs in _init_executor" and "runs
once, covering the whole cluster" are the same fact there. `mp` is not like
that: each host runs its own top-level `vllm serve` process, and the
headless node's run_headless() hardcodes vanilla
vllm.v1.executor.multiproc_executor.MultiprocExecutor rather than going
through parallel_config.distributed_executor_backend, so
TpuMultiprocExecutor._init_executor() never even runs there. Putting these
calls in check_and_update_config instead works because that hook fires
during VllmConfig construction identically on every node (leader and
headless), regardless of which executor class each one ends up
instantiating.
"""

import os
import time
from datetime import timedelta

import torch.distributed as dist
from vllm.utils.network_utils import get_ip

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_DONE_SENTINEL = "_TORCH_TPU_MP_MULTIHOST_ENV_PREPARED"
_DP_DONE_SENTINEL = "_TORCH_TPU_MP_MULTIHOST_DP_ENV_PREPARED"
_RENDEZVOUS_TIMEOUT_S = 300

# The master's TCPStore server thread stops as soon as the TCPStore object is
# garbage-collected. Other hosts may still be reading keys from it after this
# process's rendezvous call returns (there's no signal for "everyone is
# done"), so keep the master's store alive for the process lifetime instead
# of letting it go out of scope.
_keepalive_stores: list = []


def _rendezvous_port(base_port: int) -> int:
    # A dedicated port next to the caller's own coordination port so this
    # doesn't collide with vLLM's own rendezvous on the same address.
    override = os.environ.get("TORCH_TPU_MP_RENDEZVOUS_PORT")
    if override:
        return int(override)
    return int(base_port) + 1


def _rendezvous_host_ips(*, num_peers: int, peer_index: int, master_addr: str,
                         port: int, key_prefix: str) -> tuple[list[str], str]:
    """Exchange every peer's IP (and a shared xprof session id) via TCPStore.

    Returns (host_ips ordered by peer_index, xprof_session_id).
    """
    is_master = peer_index == 0
    store = dist.TCPStore(
        host_name=master_addr,
        port=port,
        world_size=num_peers,
        is_master=is_master,
        timeout=timedelta(seconds=_RENDEZVOUS_TIMEOUT_S),
    )
    if is_master:
        # Keep the server thread alive; see module-level comment.
        _keepalive_stores.append(store)

    store.set(f"{key_prefix}_ip_{peer_index}", get_ip())
    if is_master:
        store.set(f"{key_prefix}_xprof_session_id", str(time.time_ns()))

    ip_keys = [f"{key_prefix}_ip_{i}" for i in range(num_peers)]
    wait_keys = ip_keys + [f"{key_prefix}_xprof_session_id"]
    store.wait(wait_keys, timedelta(seconds=_RENDEZVOUS_TIMEOUT_S))
    host_ips = [store.get(key).decode() for key in ip_keys]
    xprof_session_id = store.get(f"{key_prefix}_xprof_session_id").decode()
    return host_ips, xprof_session_id


def prepare_mp_multihost_env(parallel_config) -> None:
    """Set TORCH_TPU_* env vars for a genuine multi-host `mp` backend run.

    No-op unless `--nnodes` > 1. Idempotent per-process so it is safe to call
    more than once (e.g. if check_and_update_config runs again for a
    per-engine config copy). See the module docstring for why this is
    called from check_and_update_config rather than _init_executor.
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

    host_ips, xprof_session_id = _rendezvous_host_ips(
        num_peers=nnodes,
        peer_index=node_rank,
        master_addr=master_addr,
        port=rendezvous_port,
        key_prefix="tpu_mp_nnodes")

    # Same construction as ray_distributed_executor.py: one address per chip,
    # ordered by (host, local_rank), spanning every host in the slice.
    base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", 8070))
    sb_addresses = [
        f"{ip}:{base_port + local_rank}" for ip in host_ips
        for local_rank in range(local_world_size)
    ]
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(sb_addresses)

    # Lazy import: tpu_platform.py imports this module from inside
    # check_and_update_config to avoid a top-level circular import. Mirrors
    # the lookup ray_distributed_executor.py uses for the Ray backend.
    from vllm_torchtpu.platforms.tpu_platform import get_tpu_multihost_topology
    topology = get_tpu_multihost_topology(total_chips)
    os.environ["TORCH_TPU_TOPOLOGY"] = topology
    os.environ["TPU_NUM_HOSTS"] = str(nnodes)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id
    os.environ[_DONE_SENTINEL] = str(nnodes)

    logger.info(
        "TPU mp multihost: node_rank=%d host_ips=%s "
        "TORCH_TPU_SLICEBUILDER_ADDRESSES=%s TORCH_TPU_TOPOLOGY=%s", node_rank,
        host_ips, os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"], topology)


def prepare_mp_multihost_dp_env(parallel_config, total_chips: int) -> bool:
    """Set TORCH_TPU_* env vars for genuine multi-host DP without Ray.

    vLLM's native multi-node DP launch (`--data-parallel-size` wider than
    `--data-parallel-size-local`, secondary hosts passing
    `--data-parallel-start-rank`) is a separate multi-node mechanism from
    `--nnodes` -- see the module docstring. check_and_update_config's
    single-host DP branch otherwise sizes the TorchTPU bootstrap by the
    *global* data_parallel_size via _prepare_singlehost_tpu_env, which
    assumes every DP replica lives on this one host; that breaks the moment
    a host only owns a slice of the DP ranks.

    Returns False (no-op) unless this host only owns a subset of the DP
    replicas, so the caller can fall back to _prepare_singlehost_tpu_env for
    the genuinely-single-host case. Idempotent per-process.
    """
    dp_size = parallel_config.data_parallel_size
    dp_size_local = parallel_config.data_parallel_size_local
    if dp_size_local <= 0 or dp_size_local >= dp_size:
        return False
    if dp_size % dp_size_local != 0:
        raise ValueError(
            f"data_parallel_size ({dp_size}) must be a multiple of "
            f"data_parallel_size_local ({dp_size_local}) for multi-host DP.")

    # Each host's `vllm serve` process picks its own random DP-init ports via
    # ParallelConfig.__post_init__ -> get_open_ports_list():
    # https://github.com/vllm-project/vllm/blob/bb4b448f9896cf3f5607bd5ed4b35b29cd866a6b/vllm/config/parallel.py#L881
    # With no coordination between hosts, those picks can disagree and hang
    # vLLM's DP-wide collectives. This is a known upstream race, not
    # TPU-specific: https://github.com/vllm-project/vllm/issues/28498.
    # Work around it by overwriting the list with ports computed identically
    # on every host from data_parallel_rpc_port (already CLI-identical), so
    # every host agrees without needing coordination of its own.
    dp_rpc_port = parallel_config.data_parallel_rpc_port
    parallel_config._data_parallel_master_port_list = [
        dp_rpc_port + 100 + i for i in range(5)
    ]

    if os.environ.get(_DP_DONE_SENTINEL) == str(dp_size):
        return True

    num_hosts = dp_size // dp_size_local
    # data_parallel_rank on this top-level config is data_parallel_start_rank
    # (0 on the head node, --data-parallel-start-rank on secondary nodes),
    # i.e. the first global DP rank this host owns.
    node_index = parallel_config.data_parallel_rank // dp_size_local
    master_addr = parallel_config.data_parallel_master_ip
    rendezvous_port = _rendezvous_port(parallel_config.data_parallel_rpc_port)

    logger.info(
        "TPU mp multihost DP: rendezvousing node_index=%d/%d "
        "data_parallel_size_local=%d master_addr=%s rendezvous_port=%d",
        node_index, num_hosts, dp_size_local, master_addr, rendezvous_port)

    host_ips, xprof_session_id = _rendezvous_host_ips(num_peers=num_hosts,
                                                      peer_index=node_index,
                                                      master_addr=master_addr,
                                                      port=rendezvous_port,
                                                      key_prefix="tpu_mp_dp")

    # One address per local chip on each host (world_size per DP replica x
    # data_parallel_size_local replicas here), ordered by (host, local
    # slot). This matches the global-rank order tpu_rank_binding.py computes
    # for DP workers (dp_rank * world_size + rank), since data_parallel_rank
    # increases in host-major order under --data-parallel-start-rank.
    world_size = parallel_config.world_size
    local_chip_count = world_size * dp_size_local
    base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", 8070))
    sb_addresses = [
        f"{ip}:{base_port + local_slot}" for ip in host_ips
        for local_slot in range(local_chip_count)
    ]
    os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(sb_addresses)

    from vllm_torchtpu.platforms.tpu_platform import get_tpu_multihost_topology
    topology = get_tpu_multihost_topology(total_chips)
    os.environ["TORCH_TPU_TOPOLOGY"] = topology
    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id
    os.environ[_DP_DONE_SENTINEL] = str(dp_size)

    logger.info(
        "TPU mp multihost DP: node_index=%d host_ips=%s "
        "TORCH_TPU_SLICEBUILDER_ADDRESSES=%s TORCH_TPU_TOPOLOGY=%s",
        node_index, host_ips, os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"],
        topology)
    return True
