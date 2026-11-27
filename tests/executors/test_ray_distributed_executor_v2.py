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
from unittest.mock import MagicMock, patch

import pytest
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from vllm.config import CacheConfig

from vllm_torchtpu.executors import ray_distributed_executor_v2 as rde2
from vllm_torchtpu.executors.ray_distributed_executor_v2 import (
    RayDistributedExecutorV2,
    get_tpu_bundles_for_indices,
)


class MockParallelConfig:
    def __init__(self):
        self.world_size = 4
        self.tensor_parallel_size = 2
        self.pipeline_parallel_size = 1
        self.prefill_context_parallel_size = 1
        self.ray_workers_use_nsight = False
        self.placement_group = None
        self.max_parallel_loading_workers = None
        self.nnodes_within_dp = 1
        self.local_world_size = 4
        self.ray_runtime_env = {}
        self.data_parallel_rank_local = None
        self.data_parallel_master_port = 0
        self.data_parallel_master_ip = "127.0.0.1"
        self.data_parallel_size = 1
        self.data_parallel_rank = 0
        self.data_parallel_index = None


class MockVllmConfig:
    def __init__(self):
        self.parallel_config = MockParallelConfig()
        self.model_config = MagicMock()
        self.cache_config = MagicMock(spec=CacheConfig)
        self.cache_config.num_gpu_blocks_override = None
        self.lora_config = MagicMock()
        self.load_config = MagicMock()
        self.scheduler_config = MagicMock()
        self.speculative_config = MagicMock()
        self.prompt_adapter_config = MagicMock()
        self.observability_config = MagicMock()
        self.device_config = MagicMock()
        self.kv_transfer_config = None
        self.instance_id = "vllm-test-instance"


@patch("vllm.v1.executor.ray_executor_v2.RayExecutorV2.__init__", lambda x, y: None)
@patch("vllm_torchtpu.executors.ray_distributed_executor_v2.ray")
@patch("vllm_torchtpu.executors.ray_distributed_executor_v2.current_platform")
@patch(
    "vllm_torchtpu.executors.ray_distributed_executor_v2.get_ip",
    return_value="127.0.0.1",
)
@patch("vllm_torchtpu.executors.ray_distributed_executor_v2._wait_until_pg_ready")
class TestTpuRayDistributedExecutorV2:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.vllm_config = MockVllmConfig()
        self.parallel_config = self.vllm_config.parallel_config

    def test_initialize_ray_cluster_basic(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        # --- Setup mocks ---
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"

        mock_ray.is_initialized.return_value = False
        mock_ray.nodes.return_value = [{"NodeID": "node_1", "Resources": {"TPU": 4}}]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node_1"
        mock_wait_until_pg_ready.return_value = None

        mock_placement_group = MagicMock()
        mock_placement_group.bundle_specs = [{"TPU": 1.0}, {"TPU": 1.0}]
        mock_ray.util.placement_group.return_value = mock_placement_group

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config

        # --- Test ---
        executor._initialize_ray_cluster()

        # --- Assertions ---
        mock_ray.init.assert_called_once()
        assert executor.parallel_config.placement_group == mock_placement_group
        mock_ray.util.placement_group.assert_called_once()

    def test_initialize_ray_cluster_pipeline_parallelism(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        self.parallel_config.pipeline_parallel_size = 2
        self.parallel_config.world_size = 8

        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [
            {"NodeID": "node_1", "Resources": {"TPU": 4}},
            {"NodeID": "node_2", "Resources": {"TPU": 4}},
        ]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node_1"
        mock_placement_group = MagicMock()
        mock_ray.util.placement_group.return_value = mock_placement_group

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config
        executor._initialize_ray_cluster()

        mock_ray.util.placement_group.assert_called_once_with(
            [{"TPU": 4, "node:127.0.0.1": 0.001}, {"TPU": 4}], strategy="PACK"
        )

    def test_initialize_ray_cluster_reuses_existing_pg(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        mock_platform.ray_device_key = "TPU"
        existing_pg = MagicMock()

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config
        executor.parallel_config.placement_group = existing_pg

        executor._initialize_ray_cluster()

        mock_ray.util.placement_group.assert_not_called()
        assert executor.parallel_config.placement_group == existing_pg

    def test_get_actor_resource_kwargs(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        mock_platform.ray_device_key = "TPU"

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.parallel_config = self.parallel_config

        resource_kwargs = executor._get_actor_resource_kwargs()

        assert resource_kwargs == {"num_gpus": 0, "resources": {"TPU": 1.0}}

    def test_slice_host_layout_sorts_by_tpu_worker_id(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        mock_platform.ray_device_key = "TPU"
        executor = RayDistributedExecutorV2(self.vllm_config)

        # Node with IP 10.0.0.99 has worker_id "0"
        # Node with IP 10.0.0.10 has worker_id "1"
        # By IP sorting alone, 10.0.0.10 would come first.
        # But by ray.io/tpu-worker-id, 10.0.0.99 must come first!
        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_b",
                "NodeManagerAddress": "10.0.0.10",
                "Resources": {"TPU": 4.0},
                "labels": {"ray.io/tpu-worker-id": "1"},
            },
            {
                "NodeID": "node_a",
                "NodeManagerAddress": "10.0.0.99",
                "Resources": {"TPU": 4.0},
                "labels": {"ray.io/tpu-worker-id": "0"},
            },
        ]

        host_order, chips_per_host = executor._slice_host_layout("TPU")
        assert host_order == ["10.0.0.99", "10.0.0.10"]
        assert chips_per_host == {"10.0.0.99": 4, "10.0.0.10": 4}

    def test_slice_host_layout_ignores_dead_nodes(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        mock_platform.ray_device_key = "TPU"
        executor = RayDistributedExecutorV2(self.vllm_config)

        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_alive",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"TPU": 4.0},
                "labels": {"ray.io/tpu-worker-id": "0"},
                "Alive": True,
            },
            {
                "NodeID": "node_dead",
                "NodeManagerAddress": "10.0.0.2",
                "Resources": {"TPU": 4.0},
                "labels": {"ray.io/tpu-worker-id": "1"},
                "Alive": False,
            },
        ]

        host_order, chips_per_host = executor._slice_host_layout("TPU")
        assert host_order == ["10.0.0.1"]
        assert chips_per_host == {"10.0.0.1": 4}

    def test_slice_host_layout_non_contiguous_worker_id_fallback(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        mock_platform.ray_device_key = "TPU"
        executor = RayDistributedExecutorV2(self.vllm_config)

        # Worker IDs 1 and 2 (missing 0, not 0-indexed)
        # Should fallback to IP sorting: 10.0.0.10 < 10.0.0.99
        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_b",
                "NodeManagerAddress": "10.0.0.99",
                "Resources": {"TPU": 4.0},
                "labels": {"ray.io/tpu-worker-id": "1"},
            },
            {
                "NodeID": "node_a",
                "NodeManagerAddress": "10.0.0.10",
                "Resources": {"TPU": 4.0},
                "labels": {"ray.io/tpu-worker-id": "2"},
            },
        ]

        host_order, chips_per_host = executor._slice_host_layout("TPU")
        assert host_order == ["10.0.0.10", "10.0.0.99"]
        assert chips_per_host == {"10.0.0.10": 4, "10.0.0.99": 4}

    def test_slice_host_layout_missing_label_fallback(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        mock_platform.ray_device_key = "TPU"
        executor = RayDistributedExecutorV2(self.vllm_config)

        # One node has label, one doesn't. Should fallback to IP sorting.
        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_b",
                "NodeManagerAddress": "10.0.0.99",
                "Resources": {"TPU": 4.0},
                "labels": {"ray.io/tpu-worker-id": "0"},
            },
            {
                "NodeID": "node_a",
                "NodeManagerAddress": "10.0.0.10",
                "Resources": {"TPU": 4.0},
                "labels": {},
            },
        ]

        host_order, chips_per_host = executor._slice_host_layout("TPU")
        assert host_order == ["10.0.0.10", "10.0.0.99"]
        assert chips_per_host == {"10.0.0.10": 4, "10.0.0.99": 4}

    @patch(
        "vllm_torchtpu.executors.ray_distributed_executor_v2.get_driver_env_vars",
        return_value={},
    )
    @patch(
        "vllm_torchtpu.executors.ray_distributed_executor_v2.get_open_port",
        return_value=9000,
    )
    @patch("vllm_torchtpu.executors.ray_distributed_executor_v2.MessageQueue")
    @pytest.mark.parametrize("bundle_indices", ["", "1,0"])
    def test_init_executor_env_vars_propagation(
        self,
        mock_mq,
        mock_port,
        mock_driver_env,
        mock_wait_until_pg_ready,
        mock_get_ip,
        mock_platform,
        mock_ray,
        monkeypatch,
        bundle_indices,
    ):
        # This test verifies that _init_executor correctly computes and propagates all
        # TPU-specific multi-host environment variables to the remote workers.

        monkeypatch.setenv("VLLM_RAY_BUNDLE_INDICES", bundle_indices)
        ordered_indices = (
            [int(i) for i in bundle_indices.split(",")] if bundle_indices else [0, 1]
        )
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_control_env_var = "TPU_VISIBLE_CHIPS"
        mock_platform._get_tpu_topology.return_value = "2x2"

        # Mock Ray cluster placement group and nodes
        mock_ray.is_initialized.return_value = True
        mock_placement_group = MagicMock()
        mock_placement_group.bundle_specs = [{"TPU": 1.0}, {"TPU": 1.0}]
        mock_ray.util.placement_group.return_value = mock_placement_group
        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_1",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"TPU": 1.0},
            },
            {
                "NodeID": "node_2",
                "NodeManagerAddress": "10.0.0.2",
                "Resources": {"TPU": 1.0},
            },
        ]

        # We have 2 workers on 2 different nodes (multi-host setup)
        bundle_to_node = [(0, "node_1", "10.0.0.1"), (1, "node_2", "10.0.0.2")]

        # Mock get_bundles_sorted_by_node or similar utils
        with (
            patch(
                "vllm_torchtpu.executors.ray_distributed_executor_v2.get_bundles_sorted_by_node",
                return_value=bundle_to_node,
            ),
            patch(
                "vllm_torchtpu.executors.ray_distributed_executor_v2.get_tpu_multihost_topology",
                return_value="2x2",
            ),
        ):
            mock_ray.get_runtime_context().get_node_id.return_value = "node_1"

            # Setup parallel config
            self.parallel_config.world_size = 2
            self.parallel_config.tensor_parallel_size = 2
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.local_world_size = 1

            executor = RayDistributedExecutorV2(self.vllm_config)
            executor.vllm_config = self.vllm_config
            executor.parallel_config = self.parallel_config

            # Mock worker actors
            mock_worker_actor_1 = MagicMock()
            mock_worker_actor_1.get_node_and_physical_gpu_ids.remote.return_value = (
                "node_1_ref"
            )
            mock_worker_actor_2 = MagicMock()
            mock_worker_actor_2.get_node_and_physical_gpu_ids.remote.return_value = (
                "node_2_ref"
            )

            # ray.remote(RayWorkerProc).options().remote() mocks
            mock_remote_class = MagicMock()
            actors = [mock_worker_actor_1, mock_worker_actor_2]
            mock_remote_class.options.return_value.remote.side_effect = [
                actors[i] for i in ordered_indices
            ]
            mock_ray.remote.return_value = mock_remote_class

            # Mock physical TPU ID discovery for each worker.
            mock_ray.get.side_effect = [
                # Discover GPU/TPU IDs (Step 6)
                [(f"node_{i + 1}", [0]) for i in ordered_indices],
                "tcp://10.0.0.1:9000",  # rank 0 reserves the TCPStore
                # Initialize workers (Step 7)
                [None, None],
                # Collect response MQ handles (Step 8)
                [
                    {"status": "READY", "handle": "handle_1"},
                    {"status": "READY", "handle": "handle_2"},
                ],
            ]

            # Run initialization
            executor._init_executor()

            excluded_vars = mock_driver_env.call_args.kwargs["worker_specific_vars"]
            assert {
                "TPU_VISIBLE_CHIPS",
                "TPU_VISIBLE_DEVICES",
                "TPU_VISIBLE_DEVICE_PATHS",
                "CLOUD_TPU_TASK_ID",
                "TPU_WORKER_ID",
                "LWS_WORKER_INDEX",
            }.issubset(excluded_vars)

            # An explicit bundle permutation can move the local worker away
            # from rank 0. The queue must use its actual rank or remote
            # workers try to attach to the driver's shared-memory queue.
            mq_args = mock_mq.call_args
            local_readers = mq_args.kwargs.get(
                "local_reader_ranks", list(range(mq_args.args[1]))
            )
            assert local_readers == [ordered_indices.index(0)]

            # Verify the environment variables passed to each worker during
            # initialize_worker.remote()
            # Worker 0 (Rank 0, Node 1)
            mock_worker_actor_1.initialize_worker.remote.assert_called_once()
            args_w0 = mock_worker_actor_1.initialize_worker.remote.call_args[0]
            local_rank_w0 = args_w0[0]
            worker_env_w0 = args_w0[1]
            kwargs_w0 = mock_worker_actor_1.initialize_worker.remote.call_args.kwargs

            assert local_rank_w0 == 0
            assert worker_env_w0["LOCAL_WORLD_SIZE"] == "1"
            assert worker_env_w0["NNODES"] == "2"
            assert worker_env_w0["NODE_RANK"] == "0"
            assert worker_env_w0["MASTER_ADDR"] == "10.0.0.1"
            assert worker_env_w0["TORCH_TPU_TOPOLOGY"] == "2x2"
            assert "10.0.0.1:8070" in worker_env_w0["TORCH_TPU_SLICEBUILDER_ADDRESSES"]
            assert "10.0.0.2:8070" in worker_env_w0["TORCH_TPU_SLICEBUILDER_ADDRESSES"]
            assert kwargs_w0["assigned_physical_gpu_ids"] == [0]

            # Worker 1 (Rank 1, Node 2)
            mock_worker_actor_2.initialize_worker.remote.assert_called_once()
            args_w1 = mock_worker_actor_2.initialize_worker.remote.call_args[0]
            local_rank_w1 = args_w1[0]
            worker_env_w1 = args_w1[1]
            kwargs_w1 = mock_worker_actor_2.initialize_worker.remote.call_args.kwargs

            assert local_rank_w1 == 0
            assert worker_env_w1["LOCAL_WORLD_SIZE"] == "1"
            assert worker_env_w1["NNODES"] == "2"
            assert worker_env_w1["NODE_RANK"] == "1"
            assert worker_env_w1["MASTER_ADDR"] == "10.0.0.1"
            assert worker_env_w1["TORCH_TPU_TOPOLOGY"] == "2x2"
            assert kwargs_w1["assigned_physical_gpu_ids"] == [0]

            executor.ray_worker_handles = []

    def test_ray_distributed_executor_v2_bundle_expansion(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        self.parallel_config.world_size = 16
        self.parallel_config.tensor_parallel_size = 16
        self.parallel_config.pipeline_parallel_size = 1

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config
        executor.world_size = 16

        # 4 host-level bundles for world_size = 16
        bundle_to_node = [
            (0, "node_0", "10.0.0.1"),
            (1, "node_1", "10.0.0.2"),
            (2, "node_2", "10.0.0.3"),
            (3, "node_3", "10.0.0.4"),
        ]

        # Test get_tpu_bundles_for_indices helper function
        selected = get_tpu_bundles_for_indices(
            None, [0, 2], bundle_to_node_id=bundle_to_node
        )
        assert selected == [(0, "node_0", "10.0.0.1"), (2, "node_2", "10.0.0.3")]
        with pytest.raises(ValueError, match="Bundle index 99 not found"):
            get_tpu_bundles_for_indices(None, [99], bundle_to_node_id=bundle_to_node)

        # Test host-level bundle expansion (4 bundles for world_size = 16)
        assignments = executor._get_bundle_assignments(None, bundle_to_node)
        assert len(assignments) == 16
        assert len(executor.bundle_assignments) == 16
        for rank in range(16):
            expected_node_id = f"node_{rank // 4}"
            assert assignments[rank]["rank"] == rank
            assert assignments[rank]["node_id"] == expected_node_id

        # Test 1:1 chip-level bundle mapping (16 bundles for world_size = 16)
        chip_bundles = [
            (i, f"node_{i // 4}", f"10.0.0.{1 + (i // 4)}") for i in range(16)
        ]
        chip_assignments = executor._get_bundle_assignments(None, chip_bundles)
        assert len(chip_assignments) == 16
        for rank in range(16):
            assert chip_assignments[rank]["rank"] == rank
            assert chip_assignments[rank]["bundle_id_idx"] == rank

        # Indivisible bundle count raises ValueError
        bad_bundle_to_node = [
            (0, "node_0", "10.0.0.1"),
            (1, "node_1", "10.0.0.2"),
            (2, "node_2", "10.0.0.3"),
        ]
        with pytest.raises(ValueError, match="divisible"):
            executor._get_bundle_assignments(None, bad_bundle_to_node)

    def test_slice_host_layout_orders_hosts_by_address(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        # Every engine derives the slice-wide view independently, so the
        # ordering has to be a function of the cluster alone.
        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_2",
                "NodeManagerAddress": "10.0.0.2",
                "Resources": {"TPU": 4.0},
            },
            {
                "NodeID": "node_1",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"TPU": 4.0},
            },
            {
                "NodeID": "node_3",
                "NodeManagerAddress": "10.0.0.3",
                "Resources": {"CPU": 8.0},
            },
        ]

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.parallel_config = self.parallel_config

        host_order, chips_per_host = executor._slice_host_layout("TPU")

        assert host_order == ["10.0.0.1", "10.0.0.2"]
        assert chips_per_host == {"10.0.0.1": 4, "10.0.0.2": 4}

    def test_data_parallel_requests_only_this_engines_bundles(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        # Under DP each engine must reserve exactly its own aligned block;
        # per-chip bundles anywhere in the slice let engines interleave.
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_1",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"TPU": 8.0},
            },
            {
                "NodeID": "node_2",
                "NodeManagerAddress": "10.0.0.2",
                "Resources": {"TPU": 8.0},
            },
        ]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node_1"
        created_pg = MagicMock()
        mock_ray.util.placement_group.return_value = created_pg

        self.parallel_config.world_size = 2
        self.parallel_config.data_parallel_size = 8
        self.parallel_config.data_parallel_index = 3

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config
        executor._initialize_ray_cluster()

        # DP rank 3 owns slice ranks 6..7, both on the first 8-chip host.
        mock_ray.util.placement_group.assert_called_once_with(
            [{"TPU": 1.0, "node:10.0.0.1": 0.001}] * 2, strategy="PACK"
        )
        assert executor.parallel_config.placement_group is created_pg

    def test_data_parallel_multi_host_block_uses_worker_id_order(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        # Hardware host order is the reverse of IP order.
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [
            {
                "NodeID": f"node_{i}",
                "NodeManagerAddress": f"10.0.0.{i}",
                "labels": {"ray.io/tpu-worker-id": str(4 - i)},
                "Resources": {"TPU": 4.0},
            }
            for i in range(1, 5)
        ]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node_1"
        created_pg = MagicMock()
        mock_ray.util.placement_group.return_value = created_pg

        self.parallel_config.world_size = 8
        self.parallel_config.data_parallel_size = 2
        self.parallel_config.data_parallel_index = 1

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config
        executor._initialize_ray_cluster()

        # DP rank 1 owns slice ranks 8..15: worker IDs 2 and 3.
        mock_ray.util.placement_group.assert_called_once_with(
            [{"TPU": 1.0, "node:10.0.0.2": 0.001}] * 4
            + [{"TPU": 1.0, "node:10.0.0.1": 0.001}] * 4,
            strategy="PACK",
        )
        assert executor.parallel_config.placement_group is created_pg

    @pytest.mark.parametrize("dp_rank", [0, 1])
    def test_data_parallel_assigns_uneven_host_overlap(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray, dp_rank
    ):
        # Each worker has its own bundle even when a block splits 8 + 4.
        mock_platform.ray_device_key = "TPU"
        mock_platform.device_name = "tpu"
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [
            {
                "NodeID": f"node_{i}",
                "NodeManagerAddress": f"10.0.0.{i}",
                "Resources": {"TPU": 8.0},
            }
            for i in range(1, 4)
        ]
        mock_ray.get_runtime_context.return_value.get_node_id.return_value = "node_1"

        self.parallel_config.world_size = 12
        self.parallel_config.data_parallel_size = 2
        self.parallel_config.data_parallel_index = dp_rank

        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.vllm_config = self.vllm_config
        executor.parallel_config = self.parallel_config

        executor.world_size = 12
        executor._initialize_ray_cluster()
        specs = mock_ray.util.placement_group.call_args.args[0]
        host_ips = [
            next(k[5:] for k in spec if k.startswith("node:")) for spec in specs
        ]
        expected = (
            ["10.0.0.1"] * 8 + ["10.0.0.2"] * 4
            if dp_rank == 0
            else ["10.0.0.2"] * 4 + ["10.0.0.3"] * 8
        )
        assert host_ips == expected
        assert all(spec["TPU"] == 1.0 for spec in specs)
        bundles = [(i, ip, ip) for i, ip in enumerate(host_ips)]
        assignments = executor._get_bundle_assignments(None, bundles)
        assert [a["bundle_id_idx"] for a in assignments] == list(range(12))

    def test_data_parallel_rejects_insufficient_host_capacity(
        self, mock_wait_until_pg_ready, mock_get_ip, mock_platform, mock_ray
    ):
        mock_platform.ray_device_key = "TPU"
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_1",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"TPU": 4.0},
            }
        ]
        self.parallel_config.world_size = 4
        self.parallel_config.data_parallel_size = 2
        self.parallel_config.data_parallel_index = 1
        executor = RayDistributedExecutorV2(self.vllm_config)
        executor.parallel_config = self.parallel_config

        with pytest.raises(ValueError, match="could reserve only 0 of 4"):
            executor._initialize_ray_cluster()
        mock_ray.util.placement_group.assert_not_called()
        mock_wait_until_pg_ready.assert_not_called()

    @pytest.mark.parametrize("bundle_indices", ["", "0,1", "1,0"])
    @patch(
        "vllm_torchtpu.executors.ray_distributed_executor_v2.get_driver_env_vars",
        return_value={},
    )
    @patch("vllm_torchtpu.executors.ray_distributed_executor_v2.MessageQueue")
    def test_init_executor_data_parallel_slice_binding(
        self,
        mock_mq,
        mock_driver_env,
        mock_wait_until_pg_ready,
        mock_get_ip,
        mock_platform,
        mock_ray,
        monkeypatch,
        bundle_indices,
    ):
        # TP=2 x DP=4 over two 4-chip hosts. This engine is DP rank 1 and Ray
        # gave it chips 2 and 3 of the first host, so its slice position comes
        # from those chip ids rather than from any cross-engine agreement.
        monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
        monkeypatch.delenv("TORCH_TPU_DP_MASTER_PORT", raising=False)
        monkeypatch.delenv("TORCH_TPU_BASE_PORT", raising=False)
        monkeypatch.setenv("VLLM_RAY_BUNDLE_INDICES", bundle_indices)

        mock_platform.ray_device_key = "TPU"
        mock_ray.is_initialized.return_value = True
        mock_placement_group = MagicMock()

        def create_pg(specs, strategy):
            mock_placement_group.bundle_specs = specs
            return mock_placement_group

        mock_ray.util.placement_group.side_effect = create_pg
        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_1",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"TPU": 4.0},
            },
            {
                "NodeID": "node_2",
                "NodeManagerAddress": "10.0.0.2",
                "Resources": {"TPU": 4.0},
            },
        ]

        def get_bundles(pg):
            return [
                (i, "node_1", next(k[5:] for k in spec if k.startswith("node:")))
                for i, spec in enumerate(pg.bundle_specs)
            ]

        with (
            patch(
                "vllm_torchtpu.executors.ray_distributed_executor_v2.get_bundles_sorted_by_node",
                side_effect=get_bundles,
            ),
            patch(
                "vllm_torchtpu.executors.ray_distributed_executor_v2.get_tpu_multihost_topology",
                return_value="1,2,2,2",
            ),
        ):
            mock_ray.get_runtime_context().get_node_id.return_value = "node_1"

            self.parallel_config.world_size = 2
            self.parallel_config.tensor_parallel_size = 2
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.local_world_size = 2
            self.parallel_config.data_parallel_size = 4
            self.parallel_config.data_parallel_index = 1

            executor = RayDistributedExecutorV2(self.vllm_config)
            executor.vllm_config = self.vllm_config
            executor.parallel_config = self.parallel_config

            actors = [MagicMock(), MagicMock()]
            mock_remote_class = MagicMock()
            mock_remote_class.options.return_value.remote.side_effect = actors
            mock_ray.remote.return_value = mock_remote_class

            mock_ray.get.side_effect = [
                [("node_1", [2]), ("node_1", [3])],
                "tcp://10.0.0.1:9000",
                [None, None],
                [
                    {"status": "READY", "handle": "handle_1"},
                    {"status": "READY", "handle": "handle_2"},
                ],
            ]

            executor._init_executor()

            expected_indices = (
                [int(i) for i in bundle_indices.split(",")]
                if bundle_indices
                else [0, 1]
            )
            assert [
                call.kwargs["scheduling_strategy"].placement_group_bundle_index
                for call in mock_remote_class.options.call_args_list
            ] == expected_indices

            actor_names = [
                call.kwargs["name"] for call in mock_remote_class.options.call_args_list
            ]
            assert all(name.endswith("_dp1") for name in actor_names)
            assert len(set(actor_names)) == 2

            # vLLM slots workers into a list sized world_size, so the
            # local_rank it is handed must stay inside this engine even though
            # the chips it got sit further along the host.
            assert [
                call[0][0]
                for call in [a.initialize_worker.remote.call_args for a in actors]
            ] == [0, 1]

            env_w0 = actors[0].initialize_worker.remote.call_args[0][1]
            env_w1 = actors[1].initialize_worker.remote.call_args[0][1]

            # Chips 2 and 3 of the first host: slice-local 2 and 3, and the
            # same slice-global ranks since this host starts the slice.
            assert env_w0["TORCH_TPU_SLICE_RANK"] == "2"
            assert env_w0["TORCH_TPU_SLICE_LOCAL_RANK"] == "2"
            assert env_w1["TORCH_TPU_SLICE_RANK"] == "3"
            assert env_w1["TORCH_TPU_SLICE_LOCAL_RANK"] == "3"
            for env in (env_w0, env_w1):
                assert env["TORCH_TPU_SLICE_WORLD_SIZE"] == "8"
                assert env["TORCH_TPU_SLICE_LOCAL_WORLD_SIZE"] == "4"
                assert env["LOCAL_WORLD_SIZE"] == "4"
                assert env["NNODES"] == "2"
                assert env["NODE_RANK"] == "0"
                # Topology and slicebuilder list cover the whole DP*TP slice,
                # derived from the cluster rather than this engine's bundles.
                assert env["TORCH_TPU_TOPOLOGY"] == "1,2,2,2"
                assert env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == (
                    "10.0.0.1:8070,10.0.0.1:8071,10.0.0.1:8072,10.0.0.1:8073,"
                    "10.0.0.2:8070,10.0.0.2:8071,10.0.0.2:8072,10.0.0.2:8073"
                )
                assert env["TORCH_TPU_DP_SIZE"] == "4"
                assert env["TORCH_TPU_DP_MASTER_ADDR"] == "10.0.0.1"
                assert env["TORCH_TPU_DP_MASTER_PORT"] == "8074"
                assert env["MASTER_ADDR"] == "10.0.0.1"
                assert env["MASTER_PORT"] == "8074"

            executor.ray_worker_handles = []

    @pytest.mark.parametrize(
        "mode",
        ["lws_worker_index_probe", "lws_worker_id_multinic", "probe_timeout"],
    )
    @patch(
        "vllm_torchtpu.executors.ray_distributed_executor_v2.get_driver_env_vars",
        return_value={},
    )
    @patch("vllm_torchtpu.executors.ray_distributed_executor_v2.MessageQueue")
    def test_init_executor_ipv6_slicebuilder_host_resolution(
        self,
        mock_mq,
        mock_driver_env,
        mock_wait_until_pg_ready,
        mock_get_ip,
        mock_platform,
        mock_ray,
        monkeypatch,
        mode,
    ):
        for key in (
            "TORCH_TPU_DP_SIZE",
            "TORCH_TPU_DP_MASTER_PORT",
            "TORCH_TPU_BASE_PORT",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES",
            "TORCH_TPU_XPROF_SESSION_ID",
            "VLLM_RAY_BUNDLE_INDICES",
            "LWS_LEADER_ADDRESS",
            "LWS_WORKER_INDEX",
        ):
            monkeypatch.delenv(key, raising=False)

        # Host 0's IPv6 literal ("2001:db8::b") sorts lexicographically after
        # host 1 ("2001:db8::2"), verifying both host ordering, layout caching
        # across _initialize_ray_cluster + _init_executor, and Step 7
        # self-published hostname substitution.
        host0_ip, host1_ip = "2001:db8::b", "2001:db8::2"
        node0_id, node1_id = "0" * 56, "1" * 56
        leader_dns = "vllm-lws-0.vllm-lws.default.svc.cluster.local"
        worker1_dns = "vllm-lws-0-1.vllm-lws.default.svc.cluster.local"

        if mode == "lws_worker_index_probe":
            monkeypatch.setenv("LWS_LEADER_ADDRESS", leader_dns)
            monkeypatch.setenv("LWS_WORKER_INDEX", "0")
            dns_to_ip = {leader_dns: host0_ip, worker1_dns: host1_ip}
            expected_hosts = (leader_dns, worker1_dns)
            remote_probe_result: object = [(1, worker1_dns)]
            nodes = [
                {
                    "NodeID": node1_id,
                    "NodeManagerAddress": host1_ip,
                    "Resources": {"TPU": 4.0},
                },
                {
                    "NodeID": node0_id,
                    "NodeManagerAddress": host0_ip,
                    "Resources": {"TPU": 4.0},
                },
            ]
        elif mode == "lws_worker_id_multinic":
            monkeypatch.setenv("LWS_LEADER_ADDRESS", leader_dns)
            monkeypatch.setenv("LWS_WORKER_INDEX", "0")
            dns_to_ip = {leader_dns: "fd00::10", worker1_dns: "fd00::11"}
            expected_hosts = (leader_dns, worker1_dns)
            remote_probe_result = [(1, worker1_dns)]
            nodes = [
                {
                    "NodeID": node1_id,
                    "NodeManagerAddress": host1_ip,
                    "NodeManagerHostname": "baremetal-node-b",
                    "Resources": {"TPU": 4.0},
                    "labels": {"ray.io/tpu-worker-id": "1"},
                },
                {
                    "NodeID": node0_id,
                    "NodeManagerAddress": host0_ip,
                    "NodeManagerHostname": "baremetal-node-a",
                    "Resources": {"TPU": 4.0},
                    "labels": {"ray.io/tpu-worker-id": "0"},
                },
            ]
        else:
            # The driver resolves its own name, but the remote probe times out:
            # startup must fail rather than guess the other host's name.
            host0_name, host1_name = "tpu-host-0.internal", "tpu-host-1.internal"
            monkeypatch.setattr(socket, "getfqdn", lambda: host0_name)
            dns_to_ip = {host0_name: host0_ip}
            expected_hosts = (host0_name, host1_name)
            remote_probe_result = TimeoutError("probe timed out")
            nodes = [
                {
                    "NodeID": node1_id,
                    "NodeManagerAddress": host1_ip,
                    "NodeManagerHostname": host1_name,
                    "Resources": {"TPU": 4.0},
                    "labels": {"ray.io/tpu-worker-id": "1"},
                },
                {
                    "NodeID": node0_id,
                    "NodeManagerAddress": host0_ip,
                    "NodeManagerHostname": host0_name,
                    "Resources": {"TPU": 4.0},
                    "labels": {"ray.io/tpu-worker-id": "0"},
                },
            ]

        monkeypatch.setattr(
            socket,
            "getnameinfo",
            lambda sockaddr, flags: (_ for _ in ()).throw(socket.gaierror("no PTR")),
        )
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, *a, **kw: (
                [
                    (
                        socket.AF_INET6,
                        socket.SOCK_STREAM,
                        6,
                        "",
                        (dns_to_ip[host], 0, 0, 0),
                    )
                ]
                if host in dns_to_ip
                else (_ for _ in ()).throw(socket.gaierror(f"unknown {host}"))
            ),
        )

        mock_platform.ray_device_key = "TPU"
        mock_ray.is_initialized.return_value = True
        mock_placement_group = MagicMock()

        def create_pg(specs, strategy):
            mock_placement_group.bundle_specs = specs
            return mock_placement_group

        mock_ray.util.placement_group.side_effect = create_pg
        mock_ray.nodes.return_value = nodes

        def get_bundles(pg):
            return [
                (i, node0_id, next(k[5:] for k in spec if k.startswith("node:")))
                for i, spec in enumerate(pg.bundle_specs)
            ]

        with (
            patch(
                "vllm_torchtpu.executors.ray_distributed_executor_v2.get_bundles_sorted_by_node",
                side_effect=get_bundles,
            ),
            patch(
                "vllm_torchtpu.executors.ray_distributed_executor_v2.get_tpu_multihost_topology",
                return_value="1,2,2,2",
            ),
        ):
            mock_ray.get_runtime_context().get_node_id.return_value = node0_id

            self.parallel_config.world_size = 2
            self.parallel_config.tensor_parallel_size = 2
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.local_world_size = 2
            self.parallel_config.data_parallel_size = 4
            self.parallel_config.data_parallel_index = 0

            executor = RayDistributedExecutorV2(self.vllm_config)
            executor.vllm_config = self.vllm_config
            executor.parallel_config = self.parallel_config

            probe_remote = MagicMock()
            probe_remote.options.return_value.remote.return_value = "probe_ref"
            actors = [MagicMock(), MagicMock()]
            actor_remote = MagicMock()
            actor_remote.options.return_value.remote.side_effect = actors

            def _ray_remote(target):
                if callable(target) and getattr(target, "__name__", "") == "_probe":
                    return probe_remote
                return actor_remote

            mock_ray.remote.side_effect = _ray_remote
            mock_ray.get.side_effect = [
                remote_probe_result,
                [(node0_id, [0]), (node0_id, [1])],
                f"tcp://[{host0_ip}]:9000",
                [None, None],
                [
                    {"status": "READY", "handle": "handle_1"},
                    {"status": "READY", "handle": "handle_2"},
                ],
            ]

            try:
                if mode == "probe_timeout":
                    with pytest.raises(RuntimeError, match="timed out"):
                        executor._init_executor()
                    mock_ray.util.placement_group.assert_not_called()
                    return
                executor._init_executor()

                # Probe ran exactly once (cached between _initialize_ray_cluster
                # and Step 7).
                assert probe_remote.options.call_count == 1
                expected_sb = ",".join(
                    f"{host}:{8070 + chip}"
                    for host in expected_hosts
                    for chip in range(4)
                )
                for actor in actors:
                    env = actor.initialize_worker.remote.call_args[0][1]
                    assert env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == expected_sb
                    assert env["MASTER_ADDR"] == host0_ip
            finally:
                executor.ray_worker_handles = []

    @patch(
        "vllm_torchtpu.executors.ray_distributed_executor_v2.get_driver_env_vars",
        return_value={},
    )
    @patch("vllm_torchtpu.executors.ray_distributed_executor_v2.MessageQueue")
    def test_data_parallel_rejects_misaligned_chip_block(
        self,
        mock_mq,
        mock_driver_env,
        mock_wait_until_pg_ready,
        mock_get_ip,
        mock_platform,
        mock_ray,
        monkeypatch,
    ):
        # Same geometry as above, but Ray hands this engine chips 1 and 2.
        # They are contiguous, so nothing downstream would look wrong, yet
        # vLLM reads the DP*TP world as `reshape(-1, dp, pp, pcp, tp)` and
        # would put slice rank 1 in the tensor parallel group of the engine on
        # chips 0..1 and slice rank 2 in the group of the engine on chips 2..3.
        # Two engines' activations would be reduced together silently, so the
        # executor has to refuse the placement instead of starting on it.
        monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
        monkeypatch.delenv("TORCH_TPU_DP_MASTER_PORT", raising=False)
        monkeypatch.delenv("TORCH_TPU_BASE_PORT", raising=False)

        mock_platform.ray_device_key = "TPU"
        mock_ray.is_initialized.return_value = True
        mock_placement_group = MagicMock()
        mock_placement_group.bundle_specs = [{"TPU": 1.0}] * 2
        mock_ray.nodes.return_value = [
            {
                "NodeID": "node_1",
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"TPU": 4.0},
            },
            {
                "NodeID": "node_2",
                "NodeManagerAddress": "10.0.0.2",
                "Resources": {"TPU": 4.0},
            },
        ]

        bundle_to_node = [(0, "node_1", "10.0.0.1"), (1, "node_1", "10.0.0.1")]

        with (
            patch(
                "vllm_torchtpu.executors.ray_distributed_executor_v2.get_bundles_sorted_by_node",
                return_value=bundle_to_node,
            ),
            patch(
                "vllm_torchtpu.executors.ray_distributed_executor_v2.get_tpu_multihost_topology",
                return_value="1,2,2,2",
            ),
        ):
            mock_ray.get_runtime_context().get_node_id.return_value = "node_1"

            self.parallel_config.world_size = 2
            self.parallel_config.tensor_parallel_size = 2
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.local_world_size = 2
            self.parallel_config.data_parallel_size = 4
            self.parallel_config.data_parallel_index = 1
            self.parallel_config.placement_group = mock_placement_group

            executor = RayDistributedExecutorV2(self.vllm_config)
            executor.vllm_config = self.vllm_config
            executor.parallel_config = self.parallel_config

            actors = [MagicMock(), MagicMock()]
            mock_remote_class = MagicMock()
            mock_remote_class.options.return_value.remote.side_effect = actors
            mock_ray.remote.return_value = mock_remote_class

            mock_ray.get.side_effect = [
                [("node_1", [1]), ("node_1", [2])],
            ]

            with pytest.raises(RuntimeError, match="aligned block"):
                executor._init_executor()

            # The refusal has to land before any worker is told to bind a
            # chip, otherwise half the slice is already up.
            for actor in actors:
                actor.initialize_worker.remote.assert_not_called()

            executor.ray_worker_handles = []


def test_ray_distributed_executor_bundle_expansion():
    """Verify V1 executor bundle index expansion for host-level and
    chip-level bundles.
    """

    def expand_bundles(bundle_indices, world_size):
        if len(bundle_indices) < world_size:
            if len(bundle_indices) == 0 or world_size % len(bundle_indices) != 0:
                raise ValueError(
                    f"world_size ({world_size}) must be divisible by "
                    f"the number of placement group bundles ({len(bundle_indices)})."
                )
            workers_per_bundle = world_size // len(bundle_indices)
            expanded_indices = []
            for b_id in bundle_indices:
                expanded_indices.extend([b_id] * workers_per_bundle)
            return expanded_indices
        elif len(bundle_indices) != world_size:
            raise ValueError(
                f"Number of bundle indices ({len(bundle_indices)}) must be less "
                f"than or equal to world_size ({world_size})."
            )
        return bundle_indices

    # 4 host-level bundle indices expanded 4x for world_size=16
    assert expand_bundles([0, 1, 2, 3], 16) == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
    ]

    # 16 chip-level bundle indices unchanged for world_size=16
    chip_indices = list(range(16))
    assert expand_bundles(chip_indices, 16) == chip_indices

    # Indivisible bundle count raises ValueError
    with pytest.raises(ValueError, match="must be divisible"):
        expand_bundles([0, 1, 2], 16)

    # Too many bundle indices raises ValueError
    with pytest.raises(ValueError, match="less than or equal to world_size"):
        expand_bundles(list(range(17)), 16)


def test_resolve_slice_host_layout_probes_once_with_affinity_and_timeout(monkeypatch):
    """Verify by-value probe, NodeAffinity node_id, ray.get timeout, and caching."""
    leader_dns = "vllm-lws-2.vllm-lws.default.svc.cluster.local"
    worker1_dns = "vllm-lws-2-1.vllm-lws.default.svc.cluster.local"
    worker2_dns = "vllm-lws-2-2.vllm-lws.default.svc.cluster.local"
    node0_id, node1_id, node2_id = "0" * 56, "1" * 56, "2" * 56

    monkeypatch.setenv("LWS_LEADER_ADDRESS", leader_dns)
    monkeypatch.setenv("LWS_WORKER_INDEX", "0")
    dns_map = {
        leader_dns: "2001:db8:1::9",
        worker1_dns: "2001:db8:1::a",
        worker2_dns: "2001:db8:1::1",
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

    nodes_calls = 0

    def _fake_nodes():
        nonlocal nodes_calls
        nodes_calls += 1
        return [
            {
                "NodeID": node2_id,
                "NodeManagerAddress": "fd00:1::1",
                "Resources": {"TPU": 8},
            },
            {
                "NodeID": node0_id,
                "NodeManagerAddress": "fd00:1::9",
                "Resources": {"TPU": 8},
            },
            {
                "NodeID": node1_id,
                "NodeManagerAddress": "fd00:1::a",
                "Resources": {"TPU": 8},
            },
        ]

    captured_fns = []
    probe_remote = MagicMock()
    probe_remote.options.return_value.remote.side_effect = ["ref_2", "ref_1"]

    def _fake_remote(fn):
        captured_fns.append(fn)
        return probe_remote

    ray_get_mock = MagicMock(return_value=[(2, worker2_dns), (1, worker1_dns)])

    monkeypatch.setattr(rde2.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        rde2.ray,
        "get_runtime_context",
        lambda: MagicMock(get_node_id=lambda: node0_id),
    )
    monkeypatch.setattr(rde2.ray, "nodes", _fake_nodes)
    monkeypatch.setattr(rde2.ray, "remote", _fake_remote)
    monkeypatch.setattr(rde2.ray, "get", ray_get_mock)

    executor = RayDistributedExecutorV2.__new__(RayDistributedExecutorV2)
    layout1 = executor._resolve_slice_host_layout("TPU")
    layout2 = executor._resolve_slice_host_layout("TPU")

    assert layout1 is layout2
    assert nodes_calls == 1
    assert len(captured_fns) == 1
    assert captured_fns[0].__module__ == "__main__"

    # Assert exact NodeAffinitySchedulingStrategy node IDs on each remote probe.
    probed_node_ids = [
        call.kwargs["scheduling_strategy"].node_id
        for call in probe_remote.options.call_args_list
    ]
    assert probed_node_ids == [node2_id, node1_id]
    for call in probe_remote.options.call_args_list:
        strategy = call.kwargs["scheduling_strategy"]
        assert isinstance(strategy, NodeAffinitySchedulingStrategy)
        assert strategy.soft is False

    # Assert ray.get uses the bounded probe timeout.
    ray_get_mock.assert_called_once_with(
        ["ref_2", "ref_1"], timeout=rde2._RAY_NODE_PROBE_TIMEOUT_S
    )

    assert layout1.host_order == ["fd00:1::9", "fd00:1::a", "fd00:1::1"]
    assert layout1.chips_per_host == dict.fromkeys(layout1.host_order, 8)
    assert layout1.sb_host_by_ip == {
        "fd00:1::9": leader_dns,
        "fd00:1::a": worker1_dns,
        "fd00:1::1": worker2_dns,
    }


def test_resolve_slice_host_layout_probe_timeout_and_duplicate_rejection(monkeypatch):
    """Probe timeout fails startup; duplicate IPv6 slicebuilder hosts fail."""
    node0_id, node1_id = "0" * 56, "1" * 56
    monkeypatch.setenv(
        "LWS_LEADER_ADDRESS", "vllm-lws-0.vllm-lws.default.svc.cluster.local"
    )
    monkeypatch.setenv("LWS_WORKER_INDEX", "0")
    monkeypatch.setattr(rde2.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        rde2.ray,
        "get_runtime_context",
        lambda: MagicMock(get_node_id=lambda: node0_id),
    )

    # Case 1: IPv4 LWS probe timeout raises instead of falling back to IP order,
    # so DP engines cannot derive different host orders.
    probe_remote = MagicMock()
    probe_remote.options.return_value.remote.return_value = "ref_1"
    monkeypatch.setattr(rde2.ray, "remote", lambda _fn: probe_remote)
    monkeypatch.setattr(
        rde2.ray,
        "get",
        MagicMock(side_effect=TimeoutError("raylet stuck")),
    )
    monkeypatch.setattr(
        rde2.ray,
        "nodes",
        lambda: [
            {
                "NodeID": node0_id,
                "NodeManagerAddress": "10.0.0.2",
                "Resources": {"TPU": 4},
            },
            {
                "NodeID": node1_id,
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"TPU": 4},
            },
        ],
    )
    executor = RayDistributedExecutorV2.__new__(RayDistributedExecutorV2)
    with pytest.raises(RuntimeError, match="timed out.*raylet stuck"):
        executor._resolve_slice_host_layout("TPU")

    # Case 2: Duplicate self-published IPv6 hostnames are rejected.
    monkeypatch.setattr(
        rde2,
        "resolve_local_slicebuilder_host",
        lambda _ip: "shared-host.internal",
    )
    monkeypatch.setattr(
        rde2.ray,
        "get",
        MagicMock(return_value=[(1, "shared-host.internal")]),
    )
    monkeypatch.setattr(
        rde2.ray,
        "nodes",
        lambda: [
            {
                "NodeID": node0_id,
                "NodeManagerAddress": "fd00::1",
                "Resources": {"TPU": 4},
                "labels": {"ray.io/tpu-worker-id": "0"},
            },
            {
                "NodeID": node1_id,
                "NodeManagerAddress": "fd00::2",
                "Resources": {"TPU": 4},
                "labels": {"ray.io/tpu-worker-id": "1"},
            },
        ],
    )
    executor2 = RayDistributedExecutorV2.__new__(RayDistributedExecutorV2)
    with pytest.raises(ValueError, match="same slicebuilder host"):
        executor2._resolve_slice_host_layout("TPU")
