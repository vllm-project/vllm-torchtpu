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
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
from vllm import platforms

from vllm_torchtpu.executors import ray_distributed_executor as ray_executor


@pytest.fixture
def executor(monkeypatch):
    instance = object.__new__(ray_executor.RayDistributedExecutor)
    instance.parallel_config = SimpleNamespace(
        world_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        placement_group=None,
        ray_workers_use_nsight=False,
    )
    instance.vllm_config = SimpleNamespace(
        parallel_config=instance.parallel_config,
        model_config=SimpleNamespace(
            model="/cache/model",
            model_weights="gs://bucket/model",
            runner_type="generate",
        ),
        kv_transfer_config=None,
        ec_transfer_config=None,
    )
    instance.scheduler_config = SimpleNamespace(async_scheduling=True)
    instance.collective_rpc = Mock()
    instance.use_ray_spmd_worker = True
    instance.kv_output_aggregator = None
    instance.forward_dag = None
    instance.workers = []
    for key in (
        "TORCH_TPU_BASE_PORT",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_XPROF_SESSION_ID",
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE",
        "RAY_USAGE_STATS_ENABLED",
    ):
        # Track absent keys too, since the module writes os.environ directly.
        monkeypatch.setenv(key, os.environ.get(key, ""))
        monkeypatch.delenv(key)
    yield instance
    instance.forward_dag = None


@pytest.fixture
def backend(monkeypatch):
    backend = MagicMock()
    backend.is_initialized.return_value = True
    backend.get_runtime_context.return_value.get_node_id.return_value = "node0"
    backend.nodes.return_value = [
        {"NodeID": "cpu", "Resources": {"CPU": 16}},
        {"NodeID": "node0", "Resources": {"TPU": 2}},
        {"NodeID": "node1", "Resources": {"TPU": 2}},
    ]
    platform = SimpleNamespace(
        ray_device_key="TPU",
        device_name="tpu",
        additional_env_vars=[],
        device_control_id_to_physical_device_id=int,
    )
    monkeypatch.setattr(ray_executor, "ray", backend)
    monkeypatch.setattr(ray_executor, "current_platform", platform)
    monkeypatch.setattr(platforms, "current_platform", platform)
    monkeypatch.setattr(ray_executor, "get_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(ray_executor, "get_open_port", lambda: 9000)
    monkeypatch.setattr(
        ray_executor, "get_tpu_multihost_topology", Mock(return_value="1,2,2,1")
    )
    monkeypatch.setattr(ray_executor, "_wait_until_pg_ready", Mock())
    monkeypatch.setenv("VLLM_RAY_BUNDLE_INDICES", "")
    return backend


@pytest.mark.parametrize("pp_size,initialized", [(1, False), (2, True)])
def test_cluster_resources_and_driver_placement(
    executor, backend, pp_size, initialized
):
    executor.parallel_config.pipeline_parallel_size = pp_size
    backend.is_initialized.return_value = initialized
    executor._initialize_ray_cluster()
    expected = (
        [{"TPU": 1.0, "node:10.0.0.1": 0.001}, {"TPU": 1.0}, {"TPU": 1.0}, {"TPU": 1.0}]
        if pp_size == 1
        else [{"TPU": 2, "node:10.0.0.1": 0.001}, {"TPU": 2}]
    )
    backend.util.placement_group.assert_called_once_with(expected, strategy="PACK")
    assert backend.init.call_count == int(not initialized)
    pg = backend.util.placement_group.return_value
    ray_executor._wait_until_pg_ready.assert_called_once_with(pg)
    assert executor.parallel_config.placement_group is pg


def test_cluster_reuses_supplied_placement(executor, backend):
    pg = object()
    executor.parallel_config.placement_group = pg
    executor._initialize_ray_cluster()
    backend.is_initialized.assert_not_called()
    backend.util.placement_group.assert_not_called()
    assert executor.parallel_config.placement_group is pg


@pytest.mark.parametrize(
    "failure,message",
    [("platform", "does not support ray"), ("driver", "Current node has no TPU")],
)
def test_invalid_cluster_fails_before_placement(executor, backend, failure, message):
    if failure == "platform":
        ray_executor.current_platform.ray_device_key = None
    else:
        backend.get_runtime_context.return_value.get_node_id.return_value = "cpu"
    with pytest.raises(ValueError, match=message):
        executor._initialize_ray_cluster()
    backend.util.placement_group.assert_not_called()


@pytest.mark.parametrize(
    "bundle_count,world_size,message",
    [
        (0, 4, "must be divisible"),
        (3, 4, "must be divisible"),
        (5, 4, "less than or equal"),
    ],
)
def test_invalid_bundle_geometry_fails_before_actor_creation(
    executor, backend, bundle_count, world_size, message
):
    executor.parallel_config.world_size = world_size
    pg = SimpleNamespace(bundle_specs=[{"TPU": 1}] * bundle_count)
    with pytest.raises(ValueError, match=message):
        executor._init_workers_ray(pg)
    backend.remote.assert_not_called()


@pytest.mark.parametrize(
    "host_bundles,explicit_indices", [(True, False), (False, False), (True, True)]
)
def test_worker_rank_environment_and_initialization_order(
    monkeypatch, executor, backend, host_bundles, explicit_indices
):
    pg = SimpleNamespace(
        bundle_specs=(
            [{"CPU": 1}, {"TPU": 2}, {"TPU": 2}] if host_bundles else [{"TPU": 1}] * 4
        )
    )
    if explicit_indices:
        monkeypatch.setenv("VLLM_RAY_BUNDLE_INDICES", "2,1")
    scheduling = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(ray_executor, "PlacementGroupSchedulingStrategy", scheduling)
    actors = [MagicMock(name=f"worker{i}") for i in range(4)]
    backend.remote.return_value.return_value.remote.side_effect = actors
    backend.get.side_effect = [
        ["10.0.0.2", "10.0.0.1", "10.0.0.2", "10.0.0.1"],
        ("node0", [1]),
        ("node0", [0]),
        ("node1", [1]),
        ("node1", [0]),
    ]
    monkeypatch.setenv("TORCH_TPU_XPROF_SESSION_ID", "session")
    monkeypatch.setenv("COPY_ME", "configured")
    copy_env = Mock(return_value={"TORCH_TPU_SLICEBUILDER_ADDRESSES", "COPY_ME"})
    monkeypatch.setattr(ray_executor, "get_env_vars_to_copy", copy_env)
    executor._init_workers_ray(pg)
    expected_indices = (
        [2, 2, 1, 1]
        if explicit_indices
        else [1, 1, 2, 2]
        if host_bundles
        else [0, 1, 2, 3]
    )
    assert [
        c.kwargs["placement_group_bundle_index"] for c in scheduling.call_args_list
    ] == expected_indices
    assert all(
        c.kwargs["placement_group"] is pg
        and c.kwargs["placement_group_capture_child_tasks"]
        for c in scheduling.call_args_list
    )
    for invocation in backend.remote.call_args_list:
        assert invocation.kwargs["resources"] == {"TPU": 1.0}
        assert invocation.kwargs["num_gpus"] == 0
    assert executor.workers == [actors[1], actors[3], actors[0], actors[2]]
    calls = executor.collective_rpc.call_args_list
    assert [c.args[0] for c in calls] == [
        "adjust_rank",
        "update_environment_variables",
        "init_worker",
        "init_device",
        "load_model",
    ]
    assert calls[0].kwargs["args"] == ({1: 0, 3: 1, 0: 2, 2: 3},)
    addresses = "10.0.0.1:8070,10.0.0.1:8071,10.0.0.2:8070,10.0.0.2:8071"
    assert os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == addresses
    environments = calls[1].kwargs["args"][0]
    assert len(environments) == 4
    for rank, env in enumerate(environments):
        assert env == {
            "NNODES": "2",
            "NODE_RANK": str(rank // 2),
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "9000",
            "TORCH_TPU_TOPOLOGY": "1,2,2,1",
            "LOCAL_WORLD_SIZE": "2",
            "TPU_NUM_HOSTS": "2",
            "TORCH_TPU_XPROF_SESSION_ID": "session",
            "COPY_ME": "configured",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": addresses,
        }
    ray_executor.get_tpu_multihost_topology.assert_called_once_with(4)
    worker_args = calls[2].kwargs["args"][0]
    assert [args["rank"] for args in worker_args] == [0, 1, 2, 3]
    assert [args["local_rank"] for args in worker_args] == [0, 1, 0, 1]
    assert [args["is_driver_worker"] for args in worker_args] == [
        True,
        False,
        True,
        False,
    ]
    for rank, args in enumerate(worker_args):
        assert args["assigned_physical_gpu_ids"] == [0, 1]
        assert args["distributed_init_method"] == "tcp://10.0.0.1:9000"
        config = args["vllm_config"]
        if rank < 2:
            assert config is executor.vllm_config
        else:
            assert config is not executor.vllm_config
            assert config.model_config.model == "gs://bucket/model"
            assert config.model_config.model_weights is None
    assert executor.vllm_config.model_config.model == "/cache/model"
    assert executor.pp_tp_workers == [executor.workers[:2], executor.workers[2:]]


@pytest.mark.parametrize(
    "connector,runner,producer,uses_sampler",
    [
        (False, "generate", None, True),
        (True, "pooling", None, False),
        (True, "generate", True, False),
        (False, "generate", False, True),
    ],
)
def test_executor_initialization(
    monkeypatch, executor, backend, connector, runner, producer, uses_sampler
):
    executor._initialize_ray_cluster = Mock()
    executor._init_workers_ray = Mock()
    executor.vllm_config.kv_transfer_config = object() if connector else None
    executor.vllm_config.model_config.runner_type = runner
    executor.vllm_config.ec_transfer_config = (
        None if producer is None else SimpleNamespace(is_ec_producer=producer)
    )
    executor.collective_rpc.return_value = ["node0:8100", "node1:8100"]
    register = Mock()
    monkeypatch.setattr(ray_executor, "set_node_kv_ip_port", register)
    executor._init_executor()
    executor._initialize_ray_cluster.assert_called_once_with()
    executor._init_workers_ray.assert_called_once_with(None)
    assert executor.has_connector is connector
    assert executor.uses_sampler is uses_sampler
    assert executor.supports_async_scheduling()
    assert executor.use_ray_compiled_dag and executor.use_ray_spmd_worker
    assert os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] == "shm"
    assert os.environ["RAY_USAGE_STATS_ENABLED"] == "0"
    if connector:
        assert register.call_args_list == [call("node0:8100"), call("node1:8100")]
    else:
        register.assert_not_called()


@pytest.mark.parametrize("aggregate", [False, True])
def test_async_future_resolves_worker_outputs(backend, aggregate):
    workers = [MagicMock(), MagicMock()]
    backend.get.side_effect = [[11, 22], ["first", "second"] if aggregate else "first"]
    aggregator = Mock() if aggregate else None
    future = ray_executor.AsyncResultFuture("dag-ref", workers, aggregator)
    output = future.result(timeout=7)
    workers[0].execute_method.remote.assert_called_once_with(
        "get_execute_model_output", 11
    )
    workers[1].execute_method.remote.assert_called_once_with(
        "get_execute_model_output", 22
    )
    refs = [w.execute_method.remote.return_value for w in workers]
    assert backend.get.call_args_list == [
        call("dag-ref", timeout=7),
        call(refs if aggregate else refs[0], timeout=7),
    ]
    if aggregate:
        aggregator.aggregate.assert_called_once_with(["first", "second"], output_rank=0)
        assert output is aggregator.aggregate.return_value
    else:
        assert output == "first"


def test_async_dag_compiled_once(executor, backend):
    dag = Mock()
    executor._compiled_ray_dag = Mock(return_value=dag)
    scheduler, grammar = object(), object()
    for _ in range(2):
        future = executor._execute_dag(scheduler, grammar, non_block=True)
        assert isinstance(future, ray_executor.AsyncResultFuture)
        assert future.result_ids_ref is dag.execute.return_value
    executor._compiled_ray_dag.assert_called_once_with(enable_asyncio=False)
    assert dag.execute.call_args_list == [call((scheduler, grammar))] * 2
    with pytest.raises(AssertionError):
        executor._execute_dag(scheduler, grammar, non_block=False)


def test_sync_dag_delegates_to_upstream(monkeypatch, executor, backend):
    executor.scheduler_config.async_scheduling = False
    execute = Mock()
    monkeypatch.setattr(ray_executor.RayDistributedExecutorV1, "_execute_dag", execute)
    scheduler, grammar = object(), object()
    assert executor._execute_dag(scheduler, grammar, True) is execute.return_value
    execute.assert_called_once_with(scheduler, grammar, True)


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(ray_executor.RayWorkerWrapperV1, "__init__", lambda self: None)
    worker = ray_executor.RayWorkerWrapper()
    worker.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True)
    )
    worker.worker = SimpleNamespace(model_runner=Mock())
    worker.compiled_dag_cuda_device_set = False
    worker._is_last_rank = Mock(return_value=True)
    return worker


@pytest.mark.parametrize("sample,intermediate", [(True, False), (False, True)])
def test_worker_async_output_lifecycle(worker, sample, intermediate):
    scheduler, grammar, tensors, output = object(), object(), object(), object()
    runner = worker.worker.model_runner
    runner.execute_model.return_value = None if sample else output
    runner.sample_tokens.return_value = output
    inputs = (scheduler, grammar, tensors) if intermediate else (scheduler, grammar)
    result_ids = [worker.execute_model_ray(inputs) for _ in range(2)]
    assert result_ids == [1, 2]
    assert worker.compiled_dag_cuda_device_set
    assert (
        runner.execute_model.call_args_list
        == [call(scheduler, tensors if intermediate else None)] * 2
    )
    assert runner.sample_tokens.call_count == (2 if sample else 0)
    if sample:
        runner.sample_tokens.assert_called_with(grammar)
    assert worker.get_execute_model_output(2) is output
    assert worker.get_execute_model_output(1) is output
    assert worker._execute_model_outputs == {}
    with pytest.raises(AssertionError, match="No output found"):
        worker.get_execute_model_output(1)


def test_worker_materializes_async_output(worker):
    output = Mock(spec=ray_executor.AsyncTPUModelRunnerOutput)
    worker._execute_model_outputs[1] = output
    assert worker.get_execute_model_output(1) is output.get_output.return_value
    output.get_output.assert_called_once_with()


def test_worker_sync_execution_delegates(monkeypatch, worker):
    worker.vllm_config.scheduler_config.async_scheduling = False
    execute = Mock()
    monkeypatch.setattr(ray_executor.RayWorkerWrapperV1, "execute_model_ray", execute)
    inputs = (object(), object())
    assert worker.execute_model_ray(inputs) is execute.return_value
    execute.assert_called_once_with(inputs)


def test_worker_device_setup_requires_worker(worker):
    worker.worker = None
    with pytest.raises(AssertionError, match="Worker is not initialized"):
        worker.setup_device_if_necessary()
