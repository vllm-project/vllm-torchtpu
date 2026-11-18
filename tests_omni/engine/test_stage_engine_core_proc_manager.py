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
"""TPU tests for NUMA node and CPU list affinity binding on stage subprocesses."""

from __future__ import annotations

import os
from multiprocessing.queues import Queue
from typing import Any

import pytest
from vllm.config import ParallelConfig, VllmConfig
from vllm.platforms import current_platform
from vllm.utils import numa_utils
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc
from vllm_omni.engine.stage_engine_core_proc_manager import StageEngineCoreProcManager

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

QUEUE_TIMEOUT_S = 30.0


def _record_stage_affinity(**kwargs: Any) -> None:
    queue: Queue[tuple[int, int, list[int]]] = kwargs["tensor_queue"]
    queue.put(
        (
            int(kwargs["omni_stage_id"]),
            int(kwargs["omni_replica_id"]),
            sorted(os.sched_getaffinity(0)),
        )
    )


def _spawn_stage_and_read_affinity(
    *,
    stage_id: int,
    numa_bind: bool,
    cpus: list[str] | None,
) -> tuple[ParallelConfig, tuple[int, int, list[int]]]:
    vllm_config = object.__new__(VllmConfig)
    vllm_config.parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        distributed_executor_backend="uni",
        data_parallel_size=1,
        numa_bind=numa_bind,
        numa_bind_nodes=[0] if numa_bind else None,
        numa_bind_cpus=cpus if numa_bind else None,
    )
    vllm_config.shutdown_timeout = 5.0

    mp_ctx = get_mp_context()
    affinity_queue: Queue[tuple[int, int, list[int]]] = mp_ctx.Queue()
    manager = StageEngineCoreProcManager(
        local_engine_count=1,
        start_index=0,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=True,
        handshake_address="ipc:///unused-numa-affinity-test",
        executor_class=UniProcExecutor,
        log_stats=False,
        omni_stage_id=stage_id,
        tensor_queue=affinity_queue,
    )
    try:
        recorded = affinity_queue.get(timeout=QUEUE_TIMEOUT_S)
    finally:
        manager.shutdown()
        affinity_queue.close()
    return vllm_config.parallel_config, recorded


def test_numa_node_and_cpu_list_affinity_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NUMA Node and CPU List Affinity Binding.

    - TPU path: spawn ``StageEngineCoreProcManager`` worker subprocesses under
      ``TpuPlatform`` with ``numa_bind=True`` vs ``numa_bind=False``.
    - Setup check: assert ``current_platform.is_cuda_alike() is False``,
      ``get_auto_numa_nodes() is None`` on TPU, and ``ParallelConfig`` records
      ``numa_bind`` and ``numa_bind_cpus``.
    - Outcome check: assert the ``numa_bind=True`` worker pins affinity to
      ``[pinned_cpu]``, ``numa_bind=False`` keeps all parent CPUs, and parent
      ``_VLLM_INTERNAL_NUMACTL_ARGS`` is restored.
    """
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv(numa_utils._NUMACTL_ARGS_ENV, "--cpunodebind=0")
    monkeypatch.setattr(
        StageEngineCoreProc,
        "run_stage_core",
        staticmethod(_record_stage_affinity),
    )

    assert current_platform.is_cuda_alike() is False, (
        "Expected TpuPlatform.is_cuda_alike() to be False"
    )
    assert numa_utils.get_auto_numa_nodes() is None, (
        "Expected automatic GPU-to-NUMA detection to return None on TPU"
    )

    parent_cpus = sorted(os.sched_getaffinity(0))
    pinned_cpu = parent_cpus[0]

    unbound_pc, unbound_record = _spawn_stage_and_read_affinity(
        stage_id=0,
        numa_bind=False,
        cpus=None,
    )
    assert unbound_pc.numa_bind is False, (
        f"Expected unbound ParallelConfig.numa_bind=False, got {unbound_pc.numa_bind}"
    )

    bound_pc, bound_record = _spawn_stage_and_read_affinity(
        stage_id=1,
        numa_bind=True,
        cpus=[str(pinned_cpu)],
    )
    assert bound_pc.numa_bind is True, (
        f"Expected bound ParallelConfig.numa_bind=True, got {bound_pc.numa_bind}"
    )
    assert bound_pc.numa_bind_cpus == [str(pinned_cpu)], (
        f"Expected numa_bind_cpus={[str(pinned_cpu)]}, got {bound_pc.numa_bind_cpus}"
    )

    assert unbound_record == (0, 0, parent_cpus), (
        f"Unbound stage affinity {unbound_record} != expected (0, 0, {parent_cpus})"
    )
    assert bound_record == (1, 0, [pinned_cpu]), (
        f"Bound stage affinity {bound_record} != expected (1, 0, {[pinned_cpu]})"
    )
    restored_env = os.environ.get(numa_utils._NUMACTL_ARGS_ENV)
    assert restored_env == "--cpunodebind=0", (
        f"Parent {numa_utils._NUMACTL_ARGS_ENV} {restored_env!r} != '--cpunodebind=0'"
    )
