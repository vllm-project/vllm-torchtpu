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
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.executors.ray_distributed_executor_v2 import (
    RayDistributedExecutorV2,
)
from vllm_torchtpu.worker.tpu_device_order import (
    plan_ascending_device_order,
    probe_host_tpu_device_ids,
    reorder_bundles_by_host_order,
    should_use_ascending_device_order,
)
from vllm_torchtpu.worker.tpu_worker import TPUWorker


def _pci_host_pairs(base_dev_id: int) -> list[tuple[int, int]]:
    pci_offsets = (0, 1, 4, 5, 6, 7, 2, 3)
    return [(lr, base_dev_id + off) for lr, off in enumerate(pci_offsets)]


def test_should_plan_ascending_order_when_host0_is_z1_and_pci_scrambled():
    host_devices = {
        "10.0.0.10": _pci_host_pairs(8),
        "10.0.0.11": _pci_host_pairs(0),
    }

    order = plan_ascending_device_order(
        host_devices, observed_host_order=["10.0.0.10", "10.0.0.11"]
    )

    assert order is not None
    assert order.host_order == ("10.0.0.11", "10.0.0.10")
    assert order.local_rank_by_host == {
        "10.0.0.11": (0, 1, 6, 7, 2, 3, 4, 5),
        "10.0.0.10": (0, 1, 6, 7, 2, 3, 4, 5),
    }
    assert order.observed_device_ids == (
        8,
        9,
        12,
        13,
        14,
        15,
        10,
        11,
        0,
        1,
        4,
        5,
        6,
        7,
        2,
        3,
    )
    # Applying the planned host order + physical local rank must yield 0..15.
    resolved = []
    for ip in order.host_order:
        dev_by_local = dict(host_devices[ip])
        for phys_lr in order.local_rank_by_host[ip]:
            resolved.append(dev_by_local[phys_lr])
    assert resolved == list(range(16))


def test_should_plan_ascending_order_for_single_host_pci_layout():
    host_devices = {"10.0.0.1": _pci_host_pairs(0)}

    order = plan_ascending_device_order(host_devices)

    assert order is not None
    assert order.host_order == ("10.0.0.1",)
    assert order.local_rank_by_host == {"10.0.0.1": (0, 1, 6, 7, 2, 3, 4, 5)}


@pytest.mark.parametrize(
    "bad_devices",
    [
        {},
        {"10.0.0.1": []},
        # Non-contiguous device IDs interleaved across hosts cannot form blocks.
        {
            "10.0.0.1": [(0, 0), (1, 2)],
            "10.0.0.2": [(0, 1), (1, 3)],
        },
        # Duplicate device ID on one host.
        {"10.0.0.1": [(0, 0), (1, 0)]},
        # Duplicate device ID across hosts.
        {"10.0.0.1": [(0, 0)], "10.0.0.2": [(0, 0)]},
        # Missing local rank.
        {"10.0.0.1": [(0, 0), (2, 1)]},
        # Device IDs not starting at 0.
        {"10.0.0.1": [(0, 1), (1, 2)]},
    ],
)
def test_should_reject_invalid_or_interleaved_host_devices(bad_devices):
    assert plan_ascending_device_order(bad_devices) is None


def test_should_reorder_bundles_preserving_intra_host_bundle_order():
    bundles = [
        (0, "n0", "10.0.0.10"),
        (1, "n0", "10.0.0.10"),
        (2, "n1", "10.0.0.11"),
        (3, "n1", "10.0.0.11"),
    ]

    reordered = reorder_bundles_by_host_order(bundles, ("10.0.0.11", "10.0.0.10"))

    assert reordered == [
        (2, "n1", "10.0.0.11"),
        (3, "n1", "10.0.0.11"),
        (0, "n0", "10.0.0.10"),
        (1, "n0", "10.0.0.10"),
    ]


def test_should_reject_bundles_from_unknown_host():
    bundles = [(0, "n0", "10.0.0.99")]
    with pytest.raises(ValueError, match="missing from target host_order"):
        reorder_bundles_by_host_order(bundles, ("10.0.0.1",))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                mode="auto",
                world_size=16,
                dp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=16,
                has_explicit_bundle_indices=False,
                has_explicit_device_override=False,
            ),
            True,
        ),
        (
            dict(
                mode="auto",
                world_size=16,
                dp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=1,
                has_explicit_bundle_indices=False,
                has_explicit_device_override=False,
            ),
            False,
        ),
        (
            dict(
                mode="1",
                world_size=16,
                dp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=1,
                has_explicit_bundle_indices=False,
                has_explicit_device_override=False,
            ),
            True,
        ),
        (
            dict(
                mode="0",
                world_size=16,
                dp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=16,
                has_explicit_bundle_indices=False,
                has_explicit_device_override=False,
            ),
            False,
        ),
        (
            dict(
                mode="1",
                world_size=16,
                dp_size=1,
                pp_size=1,
                pcp_size=2,
                dcp_size=8,
                has_explicit_bundle_indices=False,
                has_explicit_device_override=False,
            ),
            False,
        ),
        (
            dict(
                mode="1",
                world_size=16,
                dp_size=2,
                pp_size=1,
                pcp_size=1,
                dcp_size=8,
                has_explicit_bundle_indices=False,
                has_explicit_device_override=False,
            ),
            False,
        ),
        (
            dict(
                mode="1",
                world_size=16,
                dp_size=1,
                pp_size=2,
                pcp_size=1,
                dcp_size=8,
                has_explicit_bundle_indices=False,
                has_explicit_device_override=False,
            ),
            False,
        ),
        (
            dict(
                mode="1",
                world_size=16,
                dp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=16,
                has_explicit_bundle_indices=True,
                has_explicit_device_override=False,
            ),
            False,
        ),
        (
            dict(
                mode="1",
                world_size=16,
                dp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=16,
                has_explicit_bundle_indices=False,
                has_explicit_device_override=True,
            ),
            False,
        ),
    ],
)
def test_should_gate_ascending_device_order_by_parallelism_and_mode(kwargs, expected):
    assert should_use_ascending_device_order(**kwargs) is expected


def test_should_reject_invalid_ascending_device_order_mode():
    with pytest.raises(ValueError, match="Invalid ascending device order mode"):
        should_use_ascending_device_order(
            mode="bogus",
            world_size=16,
            dp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=16,
            has_explicit_bundle_indices=False,
            has_explicit_device_override=False,
        )


@patch("vllm_torchtpu.executors.ray_distributed_executor_v2.MessageQueue")
@patch(
    "vllm_torchtpu.executors.ray_distributed_executor_v2.get_open_port",
    return_value=29500,
)
@patch(
    "vllm_torchtpu.executors.ray_distributed_executor_v2.get_driver_env_vars",
    return_value={},
)
@patch("vllm_torchtpu.executors.ray_distributed_executor_v2._wait_until_pg_ready")
@patch(
    "vllm_torchtpu.executors.ray_distributed_executor_v2.get_ip",
    return_value="10.0.0.10",
)
@patch("vllm_torchtpu.executors.ray_distributed_executor_v2.current_platform")
@patch("vllm_torchtpu.executors.ray_distributed_executor_v2.ray")
def test_should_reorder_hosts_and_propagate_physical_local_rank_under_dcp(
    mock_ray,
    mock_platform,
    mock_get_ip,
    mock_wait_pg,
    mock_driver_env,
    mock_open_port,
    mock_mq,
    monkeypatch,
):
    monkeypatch.setenv("VLLM_RAY_BUNDLE_INDICES", "")
    mock_platform.ray_device_key = "TPU"
    mock_platform.device_control_env_var = "TPU_VISIBLE_CHIPS"
    mock_platform.ray_noset_device_env_vars = ()
    mock_platform.get_worker_distributed_backend.return_value = "tpu_dist"

    mock_ray.is_initialized.return_value = True
    mock_pg = MagicMock()
    mock_pg.bundle_specs = [{"TPU": 1.0}] * 4
    mock_ray.util.placement_group.return_value = mock_pg
    # Driver host 10.0.0.10 (worker_id=0) owns physical devices [3, 2],
    # remote host 10.0.0.11 (worker_id=1) owns physical devices [1, 0].
    mock_ray.nodes.return_value = [
        {
            "NodeID": "node_driver",
            "NodeManagerAddress": "10.0.0.10",
            "Resources": {"TPU": 2.0},
            "Labels": {"ray.io/tpu-worker-id": "0"},
        },
        {
            "NodeID": "node_remote",
            "NodeManagerAddress": "10.0.0.11",
            "Resources": {"TPU": 2.0},
            "Labels": {"ray.io/tpu-worker-id": "1"},
        },
    ]
    mock_ray.get_runtime_context().get_node_id.return_value = "node_driver"
    mock_ray.util.get_node_ip_address.return_value = "10.0.0.10"

    bundle_to_node = [
        (0, "node_driver", "10.0.0.10"),
        (1, "node_driver", "10.0.0.10"),
        (2, "node_remote", "10.0.0.11"),
        (3, "node_remote", "10.0.0.11"),
    ]

    vllm_config = SimpleNamespace(
        instance_id="inst-n132",
        model_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
        cache_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        device_config=SimpleNamespace(),
        load_config=SimpleNamespace(),
        lora_config=None,
        speculative_config=None,
        observability_config=None,
        parallel_config=SimpleNamespace(
            world_size=4,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=4,
            data_parallel_size=1,
            data_parallel_index=0,
            data_parallel_rank=0,
            local_world_size=2,
            nnodes=1,
            nnodes_within_dp=1,
            node_rank=0,
            placement_group=None,
            ray_runtime_env={},
            ray_workers_use_nsight=False,
            distributed_executor_backend="ray",
        ),
        kv_transfer_config=None,
    )

    actors = [MagicMock() for _ in range(4)]
    probe_task_mock = MagicMock()
    probe_task_mock.remote.side_effect = ["probe_ref_0", "probe_ref_1"]
    worker_cls_mock = MagicMock()
    worker_cls_mock.options.return_value.remote.side_effect = actors

    def fake_ray_remote(*args, **kwargs):
        if kwargs:
            return lambda fn: probe_task_mock
        return worker_cls_mock

    mock_ray.remote.side_effect = fake_ray_remote

    # Discovery returns driver host (10.0.0.10) having devices [3, 2] and remote
    # host (10.0.0.11) having devices [1, 0].
    mock_ray.get.side_effect = [
        [[(0, 3), (1, 2)], [(0, 1), (1, 0)]],
        [
            ("node_remote", [0]),
            ("node_remote", [1]),
            ("node_driver", [0]),
            ("node_driver", [1]),
        ],
        "tcp://10.0.0.11:9100",
        [None] * 4,
        [{"status": "READY", "handle": f"h{i}"} for i in range(4)],
    ]

    with (
        patch(
            "vllm_torchtpu.executors.ray_distributed_executor_v2.get_bundles_sorted_by_node",
            return_value=bundle_to_node,
        ),
        patch(
            "vllm_torchtpu.executors.ray_distributed_executor_v2.get_tpu_multihost_topology",
            return_value="2x2",
        ),
        patch.object(RayDistributedExecutorV2, "_post_init_executor"),
        patch.object(RayDistributedExecutorV2, "start_worker_monitor"),
        patch.object(RayDistributedExecutorV2, "_get_output_rank", return_value=0),
    ):
        executor = RayDistributedExecutorV2(vllm_config)

    # Driver host moved to ranks [2, 3], so SHM local_reader_ranks must be [2, 3].
    assert mock_mq.call_args.kwargs["local_reader_ranks"] == [2, 3]
    # Rank 0 & 1 are on 10.0.0.11 (node_rank=0); rank 2 & 3 on 10.0.0.10 (node_rank=1).
    # Physical local ranks invert each host's [1, 0] and [3, 2] -> (1, 0).
    assert vllm_config.additional_config["_tpu_physical_local_rank_by_rank"] == {
        0: 1,
        1: 0,
        2: 1,
        3: 0,
    }
    expected_node_ranks = ["0", "0", "1", "1"]
    for idx, actor in enumerate(actors):
        env_vars = actor.initialize_worker.remote.call_args[0][1]
        assert env_vars["NODE_RANK"] == expected_node_ranks[idx]
        assert env_vars["MASTER_ADDR"] == "10.0.0.11"
        assert env_vars["TORCH_TPU_SLICEBUILDER_ADDRESSES"].startswith(
            "10.0.0.11:8070,10.0.0.11:8071,10.0.0.10:8070,10.0.0.10:8071"
        )
    executor.ray_worker_handles = []


def test_should_honor_tpu_physical_local_rank_in_worker_init_device(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.delenv("DEBUG_TPU_LOCAL_RANK_OFFSET", raising=False)
    monkeypatch.delenv("TPU_LOCAL_RANK_OFFSET", raising=False)

    worker = object.__new__(TPUWorker)
    worker.rank = 2
    worker.local_rank = 2
    worker.distributed_init_method = "tcp://10.0.0.11:29500"
    worker.devices = ["tpu"]
    worker.is_driver_worker = False
    worker.profile_rank = 2
    worker.profile_world_size = 16
    worker.parallel_config = SimpleNamespace(
        world_size=16,
        tensor_parallel_size=16,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=16,
        data_parallel_size=1,
        data_parallel_index=0,
        data_parallel_rank_local=0,
        data_parallel_size_local=1,
        local_world_size=8,
        enable_expert_parallel=True,
    )
    worker.vllm_config = SimpleNamespace(
        parallel_config=worker.parallel_config,
        additional_config={"_tpu_physical_local_rank_by_rank": {2: 6}},
    )

    captured_env: dict[str, str] = {}

    def fake_init_dist(**kwargs):
        captured_env["RANK"] = os.environ["RANK"]
        captured_env["LOCAL_RANK"] = os.environ["LOCAL_RANK"]
        captured_env["WORLD_SIZE"] = os.environ["WORLD_SIZE"]
        captured_env["LOCAL_WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"]

    fake_pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    with (
        patch(
            "vllm_torchtpu.worker.tpu_worker.init_distributed_environment",
            side_effect=fake_init_dist,
        ),
        patch("vllm_torchtpu.worker.tpu_worker.ensure_model_parallel_initialized"),
        patch("vllm_torchtpu.worker.tpu_worker.pcp_topology_order"),
        patch("vllm_torchtpu.worker.tpu_worker.set_current_vllm_config"),
        patch("vllm_torchtpu.worker.tpu_worker.get_pp_group", return_value=fake_pp),
        patch("vllm_torchtpu.worker.tpu_worker.TPUModelRunner"),
        patch("vllm_torchtpu.worker.tpu_worker.vllm_utils.report_usage_stats"),
        patch("vllm_torchtpu.worker.tpu_worker.dist_utils.get_node_id", return_value=0),
        patch(
            "vllm.platforms.current_platform.get_worker_distributed_backend",
            return_value="tpu_dist",
        ),
    ):
        worker.init_device()

    assert captured_env == {
        "RANK": "2",
        "LOCAL_RANK": "6",
        "WORLD_SIZE": "16",
        "LOCAL_WORLD_SIZE": "8",
    }


@pytest.mark.parametrize(
    ("arch", "phys_map", "expected_local_rank", "expected_ag_bytes"),
    [
        ("GlmMoeDsaForCausalLM", {"2": 5}, "5", 6 << 20),
        ("Gemma4ForCausalLM", {2: 5}, "5", 64 << 20),
    ],
)
def test_should_rewrite_sc_all_gather_floor_in_worker_init_device_and_accept_str_keys(
    monkeypatch,
    arch: str,
    phys_map: dict,
    expected_local_rank: str,
    expected_ag_bytes: int,
):
    import importlib

    from vllm_torchtpu import env_override
    from vllm_torchtpu import envs as tpu_envs

    with patch.dict(
        os.environ,
        {"TPU_ACCELERATOR_TYPE": "tpu7x-16", "LOCAL_WORLD_SIZE": "8"},
        clear=True,
    ):
        importlib.reload(tpu_envs)
        importlib.reload(env_override)
        # Simulate RayWorkerProc: env_override already populated 64 MiB at import.
        assert (
            "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864"
            in os.environ["LIBTPU_INIT_ARGS"]
        )

        worker = object.__new__(TPUWorker)
        worker.rank = 2
        worker.local_rank = 2
        worker.distributed_init_method = "tcp://10.0.0.11:29500"
        worker.devices = ["tpu"]
        worker.is_driver_worker = False
        worker.profile_rank = 2
        worker.profile_world_size = 16
        worker.parallel_config = SimpleNamespace(
            world_size=16,
            tensor_parallel_size=16,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=16,
            data_parallel_size=1,
            data_parallel_index=0,
            data_parallel_rank_local=0,
            data_parallel_size_local=1,
            local_world_size=8,
            enable_expert_parallel=True,
        )
        worker.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(architectures=[arch]),
            parallel_config=worker.parallel_config,
            additional_config={"_tpu_physical_local_rank_by_rank": phys_map},
        )

        observed_at_dist_init: dict[str, str] = {}

        def fake_init_dist(**kwargs):
            del kwargs
            observed_at_dist_init["LOCAL_RANK"] = os.environ["LOCAL_RANK"]
            observed_at_dist_init["LIBTPU_INIT_ARGS"] = os.environ["LIBTPU_INIT_ARGS"]

        fake_pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        with (
            patch(
                "vllm_torchtpu.worker.tpu_worker.init_distributed_environment",
                side_effect=fake_init_dist,
            ),
            patch("vllm_torchtpu.worker.tpu_worker.ensure_model_parallel_initialized"),
            patch("vllm_torchtpu.worker.tpu_worker.pcp_topology_order"),
            patch("vllm_torchtpu.worker.tpu_worker.set_current_vllm_config"),
            patch("vllm_torchtpu.worker.tpu_worker.get_pp_group", return_value=fake_pp),
            patch("vllm_torchtpu.worker.tpu_worker.TPUModelRunner"),
            patch("vllm_torchtpu.worker.tpu_worker.vllm_utils.report_usage_stats"),
            patch(
                "vllm_torchtpu.worker.tpu_worker.dist_utils.get_node_id", return_value=0
            ),
            patch(
                "vllm.platforms.current_platform.get_worker_distributed_backend",
                return_value="tpu_dist",
            ),
        ):
            worker.init_device()

        assert observed_at_dist_init["LOCAL_RANK"] == expected_local_rank
        assert (
            f"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes={expected_ag_bytes}"
            in observed_at_dist_init["LIBTPU_INIT_ARGS"]
        )


def test_should_query_host_subprocesses_and_unset_stale_chip_env(monkeypatch):
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "99")
    monkeypatch.setenv("TPU_VISIBLE_DEVICES", "77")
    fake_script = (
        "import json, os, sys; "
        "assert 'TPU_VISIBLE_CHIPS' not in os.environ; "
        "assert 'TPU_VISIBLE_DEVICES' not in os.environ; "
        "assert sys.argv[1] == 'tpu_dist'; "
        "lr = int(os.environ['LOCAL_RANK']); "
        "gid = 8 + [0, 1, 4, 5][lr]; "
        "print('VLLM_TPU_DEVICE_ORDER_PROBE ' + "
        "json.dumps({'local_rank': lr, 'device_id': gid}), flush=True)"
    )

    pairs = probe_host_tpu_device_ids(
        rank_offset=0,
        local_world_size=4,
        world_size=4,
        num_nodes=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29600,
        topology="2x2",
        slicebuilder_addresses="127.0.0.1:8090",
        backend="tpu_dist",
        xprof_session_id="devorder-test",
        probe_script=fake_script,
    )

    assert pairs == [(0, 8), (1, 9), (2, 12), (3, 13)]


def test_should_raise_when_query_subprocess_fails():
    with pytest.raises(RuntimeError, match="TPU device-order discovery failed"):
        probe_host_tpu_device_ids(
            rank_offset=0,
            local_world_size=1,
            world_size=1,
            num_nodes=1,
            node_rank=0,
            master_addr="127.0.0.1",
            master_port=29600,
            topology="1x1",
            slicebuilder_addresses="127.0.0.1:8090",
            backend="tpu_dist",
            xprof_session_id="devorder-fail",
            probe_script="import sys; print('boom'); sys.exit(2)",
        )
