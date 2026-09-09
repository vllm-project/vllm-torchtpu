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

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc

from vllm_torchtpu import (_patch_multiproc_worker_error_monitor,
                           _patch_multiproc_worker_global_rank_env)


def test_worker_error_monitor_patch_flag_disabled(monkeypatch):
    monkeypatch.setenv("TPU_ENABLE_WORKER_ERROR_MONITOR", "0")
    if hasattr(MultiprocExecutor, "_tpu_worker_error_monitor_patch"):
        delattr(MultiprocExecutor, "_tpu_worker_error_monitor_patch")
    _patch_multiproc_worker_error_monitor()
    assert not getattr(MultiprocExecutor, "_tpu_worker_error_monitor_patch",
                       False)


def test_worker_error_monitor_patch_idempotent(monkeypatch):
    monkeypatch.setenv("TPU_ENABLE_WORKER_ERROR_MONITOR", "1")
    _patch_multiproc_worker_error_monitor()
    assert getattr(MultiprocExecutor, "_tpu_worker_error_monitor_patch", False)
    _patch_multiproc_worker_error_monitor()
    assert getattr(MultiprocExecutor, "_tpu_worker_error_monitor_patch", False)


def test_worker_error_monitor_and_global_rank_env_composition(monkeypatch):
    monkeypatch.setenv("TPU_ENABLE_WORKER_ERROR_MONITOR", "1")
    _patch_multiproc_worker_global_rank_env()
    assert getattr(WorkerProc, "_tpu_global_rank_env_patch", False)
    _patch_multiproc_worker_error_monitor()
    assert getattr(MultiprocExecutor, "_tpu_worker_error_monitor_patch", False)
    assert getattr(WorkerProc, "_tpu_global_rank_env_patch", False)
