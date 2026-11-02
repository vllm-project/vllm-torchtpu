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
"""Unit tests for _patch_vllm_offload_all_ranks_kv_bytes.

The patch records the sum of every worker's KV bytes per block where the
per-worker KV cache configs pass by: the engine core's scheduler-config
derivation and the worker wrapper's config hand-off.
"""

from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.engine import core as engine_core_module
from vllm.v1.worker.worker_base import WorkerWrapperBase

from vllm_torchtpu import _patch_vllm_offload_all_ranks_kv_bytes
from vllm_torchtpu.offload import raiden_store


def _worker_config(bytes_per_block, num_blocks=2):
    return SimpleNamespace(
        num_blocks=num_blocks,
        hisparse_host_num_blocks=None,
        kv_cache_tensors=[SimpleNamespace(size=bytes_per_block * num_blocks)],
    )


def test_engine_core_records_before_deriving_the_scheduler_config():
    configs = [_worker_config(1024), _worker_config(3072)]
    with (
        patch.object(raiden_store, "_all_ranks_kv_bytes_per_block", None),
        patch.object(engine_core_module, "generate_scheduler_kv_cache_config") as orig,
    ):
        _patch_vllm_offload_all_ranks_kv_bytes()
        engine_core_module.generate_scheduler_kv_cache_config(configs)
        assert raiden_store.all_ranks_kv_bytes_per_block() == 4096
        orig.assert_called_once_with(configs)


def test_worker_wrapper_records_before_handing_over_its_config():
    _patch_vllm_offload_all_ranks_kv_bytes()
    configs = [_worker_config(1024), _worker_config(3072)]
    seen = []
    wrapper = WorkerWrapperBase.__new__(WorkerWrapperBase)
    wrapper.global_rank = 1
    wrapper.vllm_config = SimpleNamespace()
    wrapper.worker = SimpleNamespace(initialize_from_config=seen.append)
    with (
        patch.object(raiden_store, "_all_ranks_kv_bytes_per_block", None),
        patch("vllm.v1.worker.worker_base.set_current_vllm_config"),
    ):
        wrapper.initialize_from_config(configs)
        assert raiden_store.all_ranks_kv_bytes_per_block() == 4096
    assert seen == [configs[1]]


def test_patch_applies_once():
    _patch_vllm_offload_all_ranks_kv_bytes()
    generate = engine_core_module.generate_scheduler_kv_cache_config
    initialize = WorkerWrapperBase.initialize_from_config
    _patch_vllm_offload_all_ranks_kv_bytes()
    assert engine_core_module.generate_scheduler_kv_cache_config is generate
    assert WorkerWrapperBase.initialize_from_config is initialize
