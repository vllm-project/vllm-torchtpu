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
from typing import Any, Dict, List

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
                                        get_bundles_for_indices,
                                        get_bundles_sorted_by_node)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_torchtpu.distributed.utils import set_node_kv_ip_port
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.tpu_platform import TPU_MULTIHOST_TOPOLOGY_MAP

logger = init_logger(__name__)


class RayDistributedExecutorV2(RayExecutorV2):
    """Ray-based distributed executor V2 for TPU.

    Inherits from vLLM V1's RayExecutorV2, leveraging the high-performance
    MessageQueue-based control plane, while keeping TPU-specific cluster
    initialization and PyTorch TPU/XLA parallelisms.
    """

    def _get_actor_resource_kwargs(self) -> dict[str, Any]:
        """Return Ray actor resource kwargs for the TPU platform.

        For torchtpu-vllm, we always use 1 TPU per worker similar to
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
        self._initialize_ray_cluster()
        placement_group = self.parallel_config.placement_group

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). ")

        # Step 2: Build bundle assignments for worker rank placement
        # while respecting VLLM_RAY_BUNDLE_INDICES.
        if envs.VLLM_RAY_BUNDLE_INDICES:
            bundle_to_node_id = get_bundles_for_indices(
                placement_group,
                list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(","))),
                self.world_size,
            )
        else:
            bundle_to_node_id = get_bundles_sorted_by_node(placement_group)

        driver_node = ray.get_runtime_context().get_node_id()

        bundle_assignments: list[dict[str, Any]] = []
        for rank, (bundle_id_idx, node_id,
                   node_ip) in enumerate(bundle_to_node_id):
            bundle_assignments.append({
                "rank": rank,
                "bundle_id_idx": bundle_id_idx,
                "node_id": node_id,
                "node_ip": node_ip,
            })

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

        node_workers: dict[str, list[int]] = defaultdict(list)
        node_physical_tpu_ids: dict[str, list[int]] = defaultdict(list)
        for i, (node_id, physical_tpu_ids
                ) in enumerate(worker_node_and_physical_tpu_ids):
            node_workers[node_id].append(i)
            node_physical_tpu_ids[node_id].extend(physical_tpu_ids)
        for node_id, physical_tpu_ids in node_physical_tpu_ids.items():
            node_physical_tpu_ids[node_id] = sorted(physical_tpu_ids)

        # Step 7: Prepare the environment variables for TorchTPU/XLA.
        # This includes construction of slice builder addresses, topology lookup,
        # and assigning rank/local_rank for the unified multi-host slice.
        host_workers = defaultdict(list)
        for bundle in bundle_assignments:
            host_workers[bundle["node_ip"]].append(bundle)

        sb_addresses = []
        base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", 8070))
        for ip in sorted(host_workers.keys()):
            for i in range(len(host_workers[ip])):
                sb_addresses.append(f"{ip}:{base_port + i}")
        slicebuilder_addresses = ",".join(sb_addresses)
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = slicebuilder_addresses
        logger.info(
            f"RayDistributedExecutorV2 | Constructed TORCH_TPU_SLICEBUILDER_ADDRESSES: {slicebuilder_addresses}"
        )

        unique_node_ids = sorted(list(node_workers.keys()))
        node_id_to_rank = {
            node_id: r
            for r, node_id in enumerate(unique_node_ids)
        }
        num_nodes = len(unique_node_ids)
        master_addr = bundle_assignments[0]["node_ip"]
        master_port = str(get_open_port())

        topology = TPU_MULTIHOST_TOPOLOGY_MAP.get(self.world_size, None)
        if topology is None:
            raise ValueError(
                f"Cannot find topology for {self.world_size} chips. "
                f"The supported number of chips are {list(TPU_MULTIHOST_TOPOLOGY_MAP.keys())}"
            )
        xprof_session_id = os.environ.get("TORCH_TPU_XPROF_SESSION_ID",
                                          str(time.time_ns()))
        os.environ["TORCH_TPU_XPROF_SESSION_ID"] = xprof_session_id

        # Initialize workers with correct environment variables and local_rank.
        init_worker_refs = []
        for i, (node_id, _) in enumerate(worker_node_and_physical_tpu_ids):
            local_rank = node_workers[node_id].index(i)
            node_rank = node_id_to_rank[node_id]
            assigned_physical_tpu_ids = sorted(node_physical_tpu_ids[node_id])

            worker_env_vars = {
                "LOCAL_WORLD_SIZE": str(len(node_workers[node_id])),
                "NNODES": str(num_nodes),
                "NODE_RANK": str(node_rank),
                "TPU_NUM_HOSTS": str(num_nodes),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
                "TORCH_TPU_TOPOLOGY": topology,
                "TORCH_TPU_XPROF_SESSION_ID": xprof_session_id,
                "TORCH_TPU_SLICEBUILDER_ADDRESSES": slicebuilder_addresses,
            }
            self.ray_worker_handles[i].local_rank = local_rank
            # Print all environment variables that will be set on the worker
            combined_env = {**self.driver_env_vars, **worker_env_vars}
            logger.debug(
                f"RayDistributedExecutorV2 | Worker {i} (Rank {i}) environment variables: {combined_env}"
            )
            init_worker_refs.append(
                self.ray_worker_handles[i].actor.initialize_worker.remote(
                    local_rank,
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
            logger.warning("Ray is not initialized, this is mainly for test.")
            ray.init()

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
            placement_group_specs = []
            for node in nodes_with_device:
                num_devices = int(node['Resources'][device_str])
                for _ in range(num_devices):
                    placement_group_specs.append({device_str: 1.0})
        else:
            assert pp_size == len(
                nodes_with_device
            ), f"Cannot use PP across hosts, please set --pipeline-parallel-size to 1 or {len(nodes_with_device)}"
            num_devices_per_pp_rank = self.parallel_config.world_size // pp_size
            placement_group_specs = [{
                device_str: num_devices_per_pp_rank
            } for _ in range(pp_size)]

        # Bind the first bundle to the current node (vLLM engine node)
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_info = next(
            (n for n in ray.nodes() if n["NodeID"] == current_node_id), None)
        current_node_resource = (current_node_info.get("Resources", {})
                                 if current_node_info else {})
        if current_node_resource.get(device_str, 0) < 1:
            raise ValueError(
                f"Current node has no {device_str} available. "
                f"{current_node_resource=}. vLLM engine cannot start without "
                f"{device_str}. Make sure you have at least 1 {device_str} "
                f"available in a node {current_node_id=} {current_ip=}.")

        # Ensure the first bundle is created on the current node
        placement_group_specs[0][f"node:{current_ip}"] = 0.001
        logger.info(
            f"RayDistributedExecutorV2 | placement_group_specs={placement_group_specs}"
        )

        # By default, Ray packs resources as much as possible.
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
            # copy it to the engine-side config here.
            overrides = self.collective_rpc("get_num_gpu_blocks_override")
            assert len(set(overrides)) == 1
            self.vllm_config.cache_config.num_gpu_blocks_override = (
                overrides[0])

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
