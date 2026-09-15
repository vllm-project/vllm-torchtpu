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

import os
import threading
import time
import weakref
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import ray
import vllm.envs as envs
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.platforms import current_platform
from vllm.utils.network_utils import (get_distributed_init_method, get_ip,
                                      get_open_port)
from vllm.v1.executor.multiproc_executor import FutureWrapper
from vllm.v1.executor.ray_env_utils import get_driver_env_vars
from vllm.v1.executor.ray_executor_v2 import RayExecutorV2, RayWorkerHandle
from vllm.v1.executor.ray_utils import (WORKER_SPECIFIC_ENV_VARS,
                                        _wait_until_pg_ready, build_actor_name,
                                        get_bundles_sorted_by_node)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.utils import set_node_kv_ip_port
from vllm_torchtpu.executors.kv_block_override import \
    reconcile_num_gpu_blocks_override
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.tpu_platform import get_tpu_multihost_topology
from vllm_torchtpu.utils import get_dp_size
from vllm_torchtpu.worker.tpu_rank_binding import slice_binding_env


def get_tpu_bundles_for_indices(
    placement_group,
    bundle_indices: list[int],
    bundle_to_node_id: Optional[list[tuple[int, str, str]]] = None,
) -> list[tuple[int, str, str]]:
    """Return bundle metadata for the specified bundle indices."""
    if bundle_to_node_id is None:
        bundle_to_node_id = get_bundles_sorted_by_node(placement_group)
    bundle_map = {b[0]: b for b in bundle_to_node_id}
    try:
        return [bundle_map[idx] for idx in bundle_indices]
    except KeyError as e:
        raise ValueError(
            f"Bundle index {e.args[0]} not found in placement group bundles: {list(bundle_map.keys())}"
        ) from None


logger = init_logger(__name__)


class RayDistributedExecutorV2(RayExecutorV2):
    """Ray-based distributed executor V2 for TPU.

    Inherits from vLLM V1's RayExecutorV2, leveraging the high-performance
    MessageQueue-based control plane, while keeping TPU-specific cluster
    initialization and PyTorch TPU/XLA parallelisms.
    """

    def _get_dp_geometry(self) -> tuple[int, int, int]:
        """Return (dp_size, dp_rank, slice_world_size) for this engine.

        vLLM runs one executor per DP engine, each sized `world_size` =
        TP x PP x PCP. TorchTPU instead needs the whole DP*TP grid bootstrapped
        as a single slice, so the executor reasons in slice-global terms and
        carves out the window of ranks belonging to this engine.
        """
        # Read the per-engine width off ParallelConfig rather than
        # self.world_size: placement group setup needs the geometry before
        # _get_parallel_sizes() has populated the executor attribute.
        parallel_config = self.parallel_config
        world_size = parallel_config.world_size
        dp_size = get_dp_size(parallel_config)
        if dp_size <= 1:
            return 1, 0, world_size

        dp_rank = parallel_config.data_parallel_index
        if dp_rank is None:
            dp_rank = parallel_config.data_parallel_rank
        if dp_rank is None:
            raise ValueError(
                "ParallelConfig.data_parallel_index must be resolved when "
                f"data_parallel_size={dp_size}.")
        return dp_size, int(dp_rank), world_size * dp_size

    def _slice_host_layout(
            self, device_str: str) -> tuple[list[str], Dict[str, int]]:
        """Return the slice's hosts in rank order and their device counts.

        Each DP engine owns a placement group covering only its own chips, so
        the slice-wide view has to come from the cluster itself. Reading it
        from `ray.nodes()` and ordering by address gives every engine the same
        answer without any of them having to agree on anything.
        """
        hosts: list[tuple[int, str, int]] = []
        has_worker_id = True
        for node in ray.nodes():
            if not node.get("Alive", True):
                continue
            resources = node.get("Resources", {})
            if device_str not in resources:
                continue
            node_ip = node.get("NodeManagerAddress") or next(
                (key.split(":", 1)[1]
                 for key in resources if key.startswith("node:")), None)
            if node_ip is None:
                raise RuntimeError(
                    f"Ray node {node.get('NodeID')} reports {device_str} but "
                    "no address to reach it by.")

            labels = node.get("labels") or node.get("Labels") or {}
            raw_worker_id = labels.get("ray.io/tpu-worker-id")
            if raw_worker_id is not None:
                try:
                    w_id = int(raw_worker_id)
                except (ValueError, TypeError):
                    w_id = 0
                    has_worker_id = False
            else:
                w_id = 0
                has_worker_id = False

            hosts.append((w_id, node_ip, int(resources[device_str])))

        ordered_by = "IP"
        if has_worker_id and len(hosts) > 0:
            hosts.sort(key=lambda x: (x[0], x[1]))
            actual_w_ids = [w_id for w_id, _, _ in hosts]
            expected_w_ids = list(range(len(hosts)))
            if actual_w_ids != expected_w_ids:
                logger.warning(
                    "RayDistributedExecutorV2 | TPU worker IDs are not "
                    "contiguous 0-indexed: got %s, expected %s. Falling back "
                    "to IP sorting.", actual_w_ids, expected_w_ids)
                hosts.sort(key=lambda x: x[1])
            else:
                ordered_by = "ray.io/tpu-worker-id"
        else:
            if len(hosts) > 1:
                logger.warning(
                    "RayDistributedExecutorV2 | Multi-host TPU detected (%d "
                    "hosts) but 'ray.io/tpu-worker-id' label is missing on "
                    "one or more nodes. Falling back to IP sorting, which may "
                    "mismatch ICI hardware topology.", len(hosts))
            hosts.sort(key=lambda x: x[1])

        logger.info(
            "RayDistributedExecutorV2 | Resolved %d TPU slice hosts "
            "(ordered by %s): %s", len(hosts), ordered_by,
            [(w_id, ip) for w_id, ip, _ in hosts])

        return [ip for _, ip, _ in hosts], {
            ip: count
            for _, ip, count in hosts
        }

    def _get_actor_resource_kwargs(self) -> dict[str, Any]:
        """Return Ray actor resource kwargs for the TPU platform.

        For vllm-torchtpu, we always use 1 TPU per worker similar to
        Multiprocexecutor.
        """
        device_key = current_platform.ray_device_key
        return {"num_gpus": 0, "resources": {device_key: 1.0}}

    def _init_executor(self) -> None:
        """Initialize the RayDistributedExecutorV2 executor.

        NOTE(ranlihao): This method shares the exact same 10-step structure as
        vLLM's base RayExecutorV2._init_executor(). The main differences are:
          1. We use a custom, TPU-optimized placement group initialization.
          2. We construct and propagate a rich set of TorchTPU multi-host
             environment variables (e.g. RANK, LOCAL_RANK, WORLD_SIZE,
             TORCH_TPU_SLICEBUILDER_ADDRESSES, etc.) as part of the worker's
             env_vars.

        TODO(ranlihao): Eventually, we should add a hook in the upstream vLLM
        RayExecutorV2 to customize/extend the placement group creation and
        the worker env_vars. This will allow us to completely delete this
        duplicated _init_executor and reuse the upstream implementation.
        """
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback = None
        self.shutting_down = False
        self.shutdown_lock = threading.Lock()

        # Step 1: Initialize Ray cluster and retrieve placement group
        if ray is None:
            raise ImportError(
                "Using Ray backend requires installation of ray.")

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). ")

        dp_size, dp_rank, slice_world_size = self._get_dp_geometry()
        if dp_size > 1 and pcp_size > 1:
            raise NotImplementedError(
                "Prefill context parallelism is not supported together with "
                "multihost data parallelism.")

        self._initialize_ray_cluster()
        placement_group = self.parallel_config.placement_group

        # Step 2: Build bundle assignments for worker rank placement while
        # respecting VLLM_RAY_BUNDLE_INDICES. If it specifies a complete 1:1
        # rank-to-bundle mapping (len == world_size), respect it; otherwise
        # fall back to all placement group bundles sorted by node. The
        # placement group holds only this engine's bundles, so these ranks are
        # per-engine; the slice-global view is derived from the cluster in
        # Step 7.
        bundle_to_node_id = None
        if envs.VLLM_RAY_BUNDLE_INDICES:
            indices = [int(x) for x in envs.VLLM_RAY_BUNDLE_INDICES.split(",")]
            if len(indices) == self.world_size:
                bundle_to_node_id = get_tpu_bundles_for_indices(
                    placement_group, indices)
        if bundle_to_node_id is None:
            bundle_to_node_id = get_bundles_sorted_by_node(placement_group)

        bundle_assignments = self._get_bundle_assignments(
            placement_group, bundle_to_node_id)
        driver_node = ray.get_runtime_context().get_node_id()

        # Step 3: Resolve the IP for torch.distributed TCPStore.
        # The TCPStore server runs on rank 0's node, so all workers
        # must be able to reach this address.
        dist_ip = bundle_assignments[0]["node_ip"]
        port = self._select_tcpstore_port(
            self.parallel_config.data_parallel_rank_local,
            self.parallel_config.data_parallel_master_port,
        )
        distributed_init_method = get_distributed_init_method(dist_ip, port)

        # Step 4: Create broadcast MessageQueue.
        # Workers on the driver node use shared memory; the rest use TCP.
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        n_local = sum(1 for a in bundle_assignments
                      if a["node_id"] == driver_node)
        self.rpc_broadcast_mq = MessageQueue(
            self.world_size,
            n_local,
            max_chunk_bytes=max_chunk_bytes,
            connect_ip=ray.util.get_node_ip_address(),
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Step 5: Spawn RayWorkerProc actors into PG bundles.
        # Use the standard RayWorkerProc.
        from vllm.v1.executor.ray_executor_v2 import RayWorkerProc

        self.ray_worker_handles: list[RayWorkerHandle] = []
        instance_id = self.vllm_config.instance_id

        # Collect driver env vars
        self.driver_env_vars = get_driver_env_vars(
            worker_specific_vars=WORKER_SPECIFIC_ENV_VARS, )

        runtime_env = self._build_runtime_env()
        resource_kwargs = self._get_actor_resource_kwargs()

        for bundle_idx in range(self.world_size):
            bundle = bundle_assignments[bundle_idx]
            is_driver_worker = self._is_driver_worker(bundle["rank"])
            is_driver_node = bundle["node_id"] == driver_node

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle["bundle_id_idx"],
            )

            actor_name = build_actor_name(instance_id, bundle["rank"], tp_size,
                                          pp_size, pcp_size)
            if dp_size > 1:
                # Every DP engine spawns the same per-engine ranks, so the
                # upstream name is only unique once the DP rank is in it.
                actor_name = f"{actor_name}_dp{dp_rank}"

            actor = (ray.remote(RayWorkerProc).options(
                name=actor_name,
                num_cpus=0,
                **resource_kwargs,
                scheduling_strategy=scheduling_strategy,
                runtime_env=runtime_env,
            ).remote(
                vllm_config=self.vllm_config,
                rank=bundle["rank"],
                distributed_init_method=distributed_init_method,
                input_shm_handle=scheduler_output_handle,
                is_driver_worker=is_driver_worker,
                is_driver_node=is_driver_node,
            ))

            handle = RayWorkerHandle(
                actor=actor,
                rank=bundle["rank"],
                local_rank=-1,  # Set in Step 7 after TPU ID discovery
                node_id=bundle["node_id"],
                bundle_id_idx=bundle["bundle_id_idx"],
            )
            self.ray_worker_handles.append(handle)

        # Step 6: Discover physical TPU IDs assigned to each worker via Ray.
        worker_node_and_physical_tpu_ids = ray.get([
            h.actor.get_node_and_physical_gpu_ids.remote()
            for h in self.ray_worker_handles
        ])

        node_physical_tpu_ids: dict[str, list[int]] = defaultdict(list)
        # vLLM slots its per-engine workers by local_rank into a list sized
        # world_size, so that index must stay within this engine. It is a
        # different quantity from the slice-local rank computed below, which
        # addresses the chips of the whole slice on a host.
        engine_node_workers: dict[str, list[int]] = defaultdict(list)
        for i, (node_id, physical_tpu_ids
                ) in enumerate(worker_node_and_physical_tpu_ids):
            engine_node_workers[node_id].append(i)
            node_physical_tpu_ids[node_id].extend(physical_tpu_ids)
        for node_id, physical_tpu_ids in node_physical_tpu_ids.items():
            node_physical_tpu_ids[node_id] = sorted(physical_tpu_ids)

        # Step 7: Prepare the environment variables for TorchTPU/XLA.
        # This includes construction of slice builder addresses, topology
        # lookup, and assigning rank/local_rank for the unified multi-host
        # slice. All of it is derived from the slice-wide assignment so that
        # every DP engine computes an identical view and merely reads different
        # entries out of it.
        host_order, chips_per_host = self._slice_host_layout(
            current_platform.ray_device_key)
        if dp_size > 1 and sum(chips_per_host.values()) != slice_world_size:
            # Only meaningful under DP, where the slice is by definition every
            # chip in the cluster. A single engine may legitimately occupy
            # less than the cluster.
            raise ValueError(
                f"Cluster reports {sum(chips_per_host.values())} chips across "
                f"{len(host_order)} hosts, but the TPU slice needs "
                f"{slice_world_size} (world_size={self.world_size} x "
                f"data_parallel_size={dp_size}).")

        # Slice-global rank of the first chip on each host.
        host_rank_offset: dict[str, int] = {}
        offset = 0
        for ip in host_order:
            host_rank_offset[ip] = offset
            offset += chips_per_host[ip]

        base_port = tpu_envs.TORCH_TPU_BASE_PORT
        sb_addresses = [
            f"{ip}:{base_port + chip}" for ip in host_order
            for chip in range(chips_per_host[ip])
        ]
        slicebuilder_addresses = ",".join(sb_addresses)
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = slicebuilder_addresses
        logger.info(
            f"RayDistributedExecutorV2 | Constructed TORCH_TPU_SLICEBUILDER_ADDRESSES: {slicebuilder_addresses}"
        )

        node_rank_by_ip = {ip: rank for rank, ip in enumerate(host_order)}
        num_nodes = len(host_order)
        master_addr = host_order[0]
        if dp_size > 1:
            # Every worker of every DP engine joins one TorchTPU world, so the
            # rendezvous port must be agreed on without a side channel. The
            # slicebuilder ports occupy [base_port, base_port + chips per host)
            # so the first port past that range is free by construction.
            max_chips_per_host = max(chips_per_host.values())
            master_port = (os.environ.get("TORCH_TPU_DP_MASTER_PORT")
                           or str(base_port + max_chips_per_host))
        else:
            master_port = str(get_open_port())

        # The topology describes the whole DP*TP slice, not this engine's
        # window of it, so it is keyed on slice_world_size.
        topology = get_tpu_multihost_topology(slice_world_size)
        xprof_session_id = os.environ.get("TORCH_TPU_XPROF_SESSION_ID",
                                          str(time.time_ns()))
        os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id

        # Resolve every worker's slice placement before starting any of them,
        # so the whole-engine invariant below can be checked while a failure
        # is still just an exception rather than a half-initialized slice.
        slice_placements: list[tuple[int, int, int]] = []
        for i, (node_id,
                physical_ids) in enumerate(worker_node_and_physical_tpu_ids):
            assignment = bundle_assignments[i]
            if node_id != assignment["node_id"]:
                raise RuntimeError(
                    f"Worker {i} started on Ray node {node_id} but its bundle "
                    f"was placed on {assignment['node_id']}; the slice rank "
                    "to chip mapping would be wrong.")
            node_ip = assignment["node_ip"]
            # Ray hands each worker one chip; its physical index on the host is
            # what TorchTPU binds to, and it also places the worker in the
            # slice. Every engine derives this the same way from the cluster,
            # so no two DP engines can disagree about who owns which chip.
            if len(physical_ids) != 1:
                raise RuntimeError(
                    f"Worker {i} was assigned {physical_ids} chips; the TPU "
                    "executor requires exactly one chip per worker.")
            slice_local_rank = int(physical_ids[0])
            local_world_size = chips_per_host[node_ip]
            if not 0 <= slice_local_rank < local_world_size:
                raise RuntimeError(
                    f"Worker {i} got chip {slice_local_rank} on {node_ip}, "
                    f"which reports only {local_world_size} chips.")
            slice_placements.append(
                (host_rank_offset[node_ip] + slice_local_rank,
                 slice_local_rank, local_world_size))

        if dp_size > 1:
            # vLLM lays the DP*TP world out as
            # `arange(slice_world).reshape(-1, dp_size, pp, pcp, tp)`, so it
            # reads ranks [k*W, (k+1)*W) as one engine's TP group and the
            # stride-W ranks as one member per engine. Nothing enforces that
            # this engine's chips land on such a block: Ray hands out whatever
            # was free, and the slice rank follows the physical chip. The
            # engine placed on chips {1, 2} of a host would have its two
            # workers split across the TP groups of its neighbours, and every
            # collective from there on would mix two engines' activations
            # without any of them noticing. Model parallel group membership is
            # only correct on an aligned block, so require one.
            #
            # The labels may still permute -- the engine on the first block is
            # not necessarily dp_rank 0 -- but that is harmless, because both
            # expert sharding (FusedMoEParallelConfig, via
            # get_dp_group().rank_in_group) and the DP token counts
            # (uniform under EP lockstep) are read off group position rather
            # than data_parallel_rank.
            engine_ranks = sorted(rank for rank, _, _ in slice_placements)
            block_start = engine_ranks[0]
            if (block_start % self.world_size != 0 or engine_ranks != list(
                    range(block_start, block_start + self.world_size))):
                raise RuntimeError(
                    f"DP rank {dp_rank} was placed on slice ranks "
                    f"{engine_ranks}, which is not an aligned block of "
                    f"{self.world_size}. vLLM would pair these workers with "
                    "another engine's when it builds the tensor parallel "
                    "groups. Free the slice and retry so Ray can hand each "
                    "engine a contiguous, aligned set of chips.")

        # Initialize workers with correct environment variables and local_rank.
        init_worker_refs = []
        for i, (node_id,
                physical_ids) in enumerate(worker_node_and_physical_tpu_ids):
            assignment = bundle_assignments[i]
            node_ip = assignment["node_ip"]
            # Slot inside this engine, handed to vLLM.
            engine_local_rank = engine_node_workers[node_id].index(i)
            slice_rank, slice_local_rank, local_world_size = slice_placements[
                i]
            node_rank = node_rank_by_ip[node_ip]
            assigned_physical_tpu_ids = sorted(node_physical_tpu_ids[node_id])

            worker_env_vars = {
                "LOCAL_WORLD_SIZE": str(local_world_size),
                "NNODES": str(num_nodes),
                "NODE_RANK": str(node_rank),
                "TPU_NUM_HOSTS": str(num_nodes),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
                "TORCH_TPU_TOPOLOGY": topology,
                "TORCH_TPU_XPROF_SESSION_ID": xprof_session_id,
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": slicebuilder_addresses,
            }
            if dp_size > 1:
                # TPUWorker rebuilds MASTER_ADDR/PORT from the DP pair, and
                # TORCH_TPU_DP_SIZE is what makes get_dp_size() in the worker
                # agree with the slice the executor actually built.
                worker_env_vars.update({
                    "TORCH_TPU_DP_SIZE": str(dp_size),
                    "TORCH_TPU_DP_MASTER_ADDR": master_addr,
                    "TORCH_TPU_DP_MASTER_PORT": master_port,
                })
                worker_env_vars.update(
                    slice_binding_env(
                        rank=slice_rank,
                        local_rank=slice_local_rank,
                        world_size=slice_world_size,
                        local_world_size=local_world_size,
                    ))
            self.ray_worker_handles[i].local_rank = engine_local_rank
            logger.info(
                "RayDistributedExecutorV2 | Worker rank=%d dp_rank=%d "
                "engine_local_rank=%d -> slice_rank=%d slice_local_rank=%d "
                "node=%s (node_rank=%d)", i, dp_rank, engine_local_rank,
                slice_rank, slice_local_rank, node_ip, node_rank)
            # Print all environment variables that will be set on the worker
            combined_env = {**self.driver_env_vars, **worker_env_vars}
            logger.debug(
                f"RayDistributedExecutorV2 | Worker {i} (slice rank {slice_rank}) environment variables: {combined_env}"
            )
            init_worker_refs.append(
                self.ray_worker_handles[i].actor.initialize_worker.remote(
                    engine_local_rank,
                    worker_env_vars,
                    self.driver_env_vars,
                    assigned_physical_gpu_ids=assigned_physical_tpu_ids,
                ))
        if len(node_physical_tpu_ids) == 1:
            node_id_0 = worker_node_and_physical_tpu_ids[0][0]
            self.vllm_config.parallel_config.assigned_physical_gpu_ids = sorted(
                node_physical_tpu_ids[node_id_0])
        ray.get(init_worker_refs)

        # Step 8: Collect response MQ handles
        init_results = ray.get(
            [h.actor.wait_for_init.remote() for h in self.ray_worker_handles])

        self.response_mqs: list[MessageQueue] = []
        for i, result in enumerate(init_results):
            if result["status"] != RayWorkerProc.READY_STR:
                raise RuntimeError(
                    f"Worker {i} failed to initialize: {result}")
            self.response_mqs.append(
                MessageQueue.create_from_handle(result["handle"], 0))

        # Step 9: Start run() before wait_until_ready() to avoid deadlock
        for handle in self.ray_worker_handles:
            handle.run()

        # Step 10: wait_until_ready() barrier
        self.rpc_broadcast_mq.wait_until_ready()
        for response_mq in self.response_mqs:
            response_mq.wait_until_ready()

        self.futures_queue = deque[FutureWrapper]()
        self._post_init_executor()

        self.start_worker_monitor()
        self.output_rank = self._get_output_rank()

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        if self.has_connector:
            ip_port = self.collective_rpc("get_node_kv_ip_port")
            for item in ip_port:
                set_node_kv_ip_port(item)

    def _initialize_ray_cluster(self) -> None:
        """Initialize the distributed cluster with Ray.

        Creates a TPU-optimized placement group where all chips on a node
        are packed into a single bundle, instead of 1 GPU per bundle.
        """
        # Check if placement group is already provided (e.g., by ray.serve.llm)
        if self.parallel_config.placement_group is not None:
            logger.info(
                f"Using existing placement group: {self.parallel_config.placement_group}"
            )
            return

        if ray.is_initialized():
            logger.info(
                "Ray is already initialized. Skipping Ray initialization.")
        else:
            logger.info(
                "Connecting to Ray cluster via ray.init(address='auto').")
            ray.init(address="auto", ignore_reinit_error=True)

        device_str = current_platform.ray_device_key
        if not device_str:
            raise ValueError(
                f"current platform {current_platform.device_name} does not "
                "support ray.")

        pp_size = self.parallel_config.pipeline_parallel_size
        placement_group_specs: List[Dict[str, float]] = []

        ray_nodes = ray.nodes()
        logger.info(f"RayDistributedExecutorV2 | ray_nodes={ray_nodes}")

        # Filter nodes that have the required TPU resource
        nodes_with_device = [
            n for n in ray_nodes if device_str in n.get("Resources", {})
        ]
        logger.info(
            f"RayDistributedExecutorV2 | nodes_with_device={len(nodes_with_device)} "
            f"(filtered from {len(ray_nodes)} total nodes)")

        if pp_size == 1:
            # One bundle per device of *this engine*, as upstream
            # initialize_ray_cluster() does. Sizing this by the cluster's
            # devices instead only coincides with world_size when a single
            # engine spans the whole slice; under DP every engine would ask
            # for every chip and none but the first could ever be scheduled.
            placement_group_specs = [{
                device_str: 1.0
            } for _ in range(self.parallel_config.world_size)]
        else:
            # One bundle per pipeline stage, sized to the stage's devices.
            # PACK fills a host before moving to the next, so consecutive
            # stages share a host whenever the stage size allows it.
            num_devices_per_pp_rank = self.parallel_config.world_size // pp_size
            placement_group_specs = [{
                device_str: num_devices_per_pp_rank
            } for _ in range(pp_size)]

        dp_size, _, _ = self._get_dp_geometry()
        if dp_size == 1:
            # Bind the first bundle to the current node (vLLM engine node).
            # Under DP every engine runs on the head node, so pinning would
            # crowd all of them onto it and split each engine's TP group
            # across hosts; upstream skips this for RayExecutorV2 entirely
            # (require_gpu_on_driver=False).
            current_ip = get_ip()
            current_node_id = ray.get_runtime_context().get_node_id()
            current_node_info = next(
                (n for n in ray.nodes() if n["NodeID"] == current_node_id),
                None)
            current_node_resource = (current_node_info.get("Resources", {})
                                     if current_node_info else {})
            if current_node_resource.get(device_str, 0) < 1:
                raise ValueError(
                    f"Current node has no {device_str} available. "
                    f"{current_node_resource=}. vLLM engine cannot start "
                    f"without {device_str}. Make sure you have at least 1 "
                    f"{device_str} available in a node {current_node_id=} "
                    f"{current_ip=}.")
            placement_group_specs[0][f"node:{current_ip}"] = 0.001

        logger.info(
            f"RayDistributedExecutorV2 | placement_group_specs={placement_group_specs}"
        )

        # By default, Ray packs resources as much as possible. Each engine
        # owns its own group; Ray's resource accounting is what keeps DP
        # engines on disjoint chips.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

        assert current_placement_group is not None
        self.parallel_config.placement_group = current_placement_group

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        specs = super().get_kv_cache_specs()

        if self.vllm_config.cache_config.num_gpu_blocks_override is None:
            # Compact-mamba sizing sets `num_gpu_blocks_override` on the worker's
            # cache_config during the RPC above; workers are separate processes, so
            # copy it to the engine-side config here. Tolerate per-worker HBM
            # measurement jitter exactly like TpuMultiprocExecutor: build 155
            # died here on a 3-block spread (6456..6459 at TP=32) because this
            # path still demanded exact agreement.
            pc = self.vllm_config.parallel_config
            agreed = reconcile_num_gpu_blocks_override(
                self.collective_rpc("get_num_gpu_blocks_override"),
                workers_per_stage=pc.world_size // pc.pipeline_parallel_size)
            if agreed is not None:
                self.vllm_config.cache_config.num_gpu_blocks_override = agreed

        return specs

    def initialize_from_config(self,
                               kv_cache_configs: list[KVCacheConfig]) -> None:
        super().initialize_from_config(kv_cache_configs)
        # Per-shape H2D prewarm: each shape is its own collective_rpc call with
        # its own 60-second shm_broadcast window.  Workers that have no KV
        # offload spec return [] from get_kv_prewarm_shapes() — a no-op.
        shapes = self.collective_rpc("get_kv_prewarm_shapes")
        if not shapes or not shapes[0]:
            return
        for p in shapes[0]:
            self.collective_rpc("prewarm_kv_offload_shape", args=(p, ))

    def _get_bundle_assignments(
        self,
        placement_group,
        bundle_to_node_id: list[tuple[int, str, str]],
    ) -> list[dict[str, Any]]:
        num_bundles = len(bundle_to_node_id)
        if num_bundles == 0 or self.world_size % num_bundles != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"the number of placement group bundles ({num_bundles}).")
        workers_per_bundle = self.world_size // num_bundles
        bundle_assignments: list[dict[str, Any]] = []
        for rank in range(self.world_size):
            bundle_idx = rank // workers_per_bundle
            bundle_id_idx, node_id, node_ip = bundle_to_node_id[bundle_idx]
            bundle_assignments.append({
                "rank": rank,
                "bundle_id_idx": bundle_id_idx,
                "node_id": node_id,
                "node_ip": node_ip,
            })
        self.bundle_assignments = bundle_assignments
        return bundle_assignments
