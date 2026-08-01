# Copyright 2025 Google LLC
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

from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CacheConfig
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from vllm_torchtpu.executors.tpu_multiproc_executor import TpuMultiprocExecutor

# Sentinel returned by the parent get_kv_cache_specs; the override plumbing must
# pass this through unchanged.
_SPECS = [{"layer": MagicMock()}]


def _make_executor(engine_override, worker_overrides):
    """Build an executor without running __init__, with a mocked engine
    cache_config and a collective_rpc that returns the per-worker overrides."""
    executor = TpuMultiprocExecutor.__new__(TpuMultiprocExecutor)
    executor.vllm_config = MagicMock()
    executor.vllm_config.cache_config = MagicMock(spec=CacheConfig)
    executor.vllm_config.cache_config.num_gpu_blocks_override = engine_override
    executor.collective_rpc = MagicMock(return_value=worker_overrides)
    return executor


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_copies_override_from_workers_when_engine_unset(mock_super):
    """Engine override is None -> copy the agreed worker value to the engine."""
    executor = _make_executor(engine_override=None,
                              worker_overrides=[5428, 5428, 5428])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    executor.collective_rpc.assert_called_once_with(
        "get_num_gpu_blocks_override")
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 5428


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_does_not_overwrite_existing_engine_override(mock_super):
    """A user-supplied engine override (e.g. --num-gpu-blocks-override) wins;
    the worker value is never queried."""
    executor = _make_executor(engine_override=1234,
                              worker_overrides=[5428, 5428, 5428])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    executor.collective_rpc.assert_not_called()
    assert executor.vllm_config.cache_config.num_gpu_blocks_override == 1234


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_raises_when_workers_disagree(mock_super):
    """Workers must agree on the override; a mismatch is a sizing bug, not
    something to silently pick one of."""
    executor = _make_executor(engine_override=None,
                              worker_overrides=[5428, 6840, 5428])

    with pytest.raises(AssertionError):
        executor.get_kv_cache_specs()


@patch.object(MultiprocExecutor, "get_kv_cache_specs", return_value=_SPECS)
def test_no_override_when_workers_return_none(mock_super):
    """If no worker computed an override (uniform sizing skipped), the engine
    stays None and vLLM falls back to its own num_blocks computation."""
    executor = _make_executor(engine_override=None,
                              worker_overrides=[None, None, None])

    specs = executor.get_kv_cache_specs()

    assert specs is _SPECS
    assert executor.vllm_config.cache_config.num_gpu_blocks_override is None
